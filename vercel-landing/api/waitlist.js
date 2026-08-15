/**
 * Public waitlist intake for trybuddynow.com — Vercel serverless port of
 * landing/steps/service/waitlist.py (which ran behind Caddy on ark.mit.edu).
 *
 * Keeps the original guarantees: the Supabase key never reaches the browser,
 * the endpoint never reads back (204 on success AND duplicate, so there is no
 * email-enumeration signal), and there is no GET.
 *
 * Deliberately zero-dependency (built-in fetch only) so the deploy stays a
 * pure static upload with no npm install and no build step. It is also NOT
 * Python: a .py file here re-triggers Vercel's FastAPI autodetection, which is
 * what broke earlier builds.
 */

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_KEY = process.env.SUPABASE_WAITLIST_KEY;

const RATE_MAX = parseInt(process.env.WAITLIST_RATE_MAX || "5", 10);
const RATE_WINDOW = parseInt(process.env.WAITLIST_RATE_WINDOW || "60", 10) * 1000;

// Sanity gate, not full RFC 5322 — same expression as the original service.
const EMAIL = /^[^@\s]{1,64}@[^@\s]{1,255}\.[^@\s]{2,}$/;

// Best-effort only. Serverless instances are ephemeral and each one keeps its
// own Map, so this blunts a naive flood from a single client but is NOT a real
// distributed limit. Use Vercel's WAF rate limiting for that.
const hits = new Map();

function rateOk(ip) {
  const now = Date.now();
  const q = (hits.get(ip) || []).filter((t) => t > now - RATE_WINDOW);
  if (q.length >= RATE_MAX) {
    hits.set(ip, q);
    return false;
  }
  q.push(now);
  hits.set(ip, q);
  if (hits.size > 5000) hits.clear(); // crude unbounded-growth guard
  return true;
}

function clientIp(req) {
  const xff = req.headers["x-forwarded-for"] || "";
  return xff ? String(xff).split(",")[0].trim() : req.socket?.remoteAddress || "?";
}

module.exports = async (req, res) => {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).end();
  }
  if (!SUPABASE_URL || !SUPABASE_KEY) {
    // Fail loudly in logs, quietly to the client — never leak config state.
    console.error("waitlist: SUPABASE_URL or SUPABASE_WAITLIST_KEY is unset");
    return res.status(500).end();
  }

  if (!rateOk(clientIp(req))) return res.status(429).end();

  const body = typeof req.body === "object" && req.body ? req.body : {};

  // Honeypot tripped: accept and silently drop, so the bot sees success.
  if (body.company_website) return res.status(204).end();

  const email = String(body.email || "").trim().toLowerCase();
  if (email.length > 254 || !EMAIL.test(email)) return res.status(400).end();

  const r = await fetch(`${SUPABASE_URL}/rest/v1/waitlist`, {
    method: "POST",
    headers: {
      apikey: SUPABASE_KEY,
      Authorization: `Bearer ${SUPABASE_KEY}`,
      "Content-Type": "application/json",
      // ignore-duplicates = ON CONFLICT DO NOTHING; return=minimal = no read-back.
      Prefer: "resolution=ignore-duplicates,return=minimal",
    },
    body: JSON.stringify({
      email,
      source: "landing",
      user_agent: String(req.headers["user-agent"] || "").slice(0, 300),
    }),
  });

  if (!r.ok && r.status !== 409) {
    console.error("waitlist: supabase insert failed", r.status, await r.text());
    return res.status(500).end();
  }
  return res.status(204).end();
};
