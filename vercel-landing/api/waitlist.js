/**
 * Public waitlist intake for trybuddynow.com — Vercel serverless port of
 * landing/steps/service/waitlist.py (which ran behind Caddy on ark.mit.edu).
 *
 * Writes straight to Supabase Postgres as the insert-only `waitlist_writer`
 * role (waitlist-setup/002), restoring the original least-privilege design.
 * Keeps the original guarantees: credentials never reach the browser, the
 * endpoint never reads back (204 on success AND duplicate, so there is no
 * email-enumeration signal), and there is no GET.
 *
 * NOT Python: a .py file here re-triggers Vercel's FastAPI autodetection,
 * which is what broke earlier builds.
 */

const { Pool } = require("pg");

// Supabase Root 2021 CA. The pooler presents a cert from Supabase's own CA, not
// a public one, so Node's default trust store rejects it. Pinning this is what
// lets us keep full verification instead of rejectUnauthorized:false — an
// unverified TLS session to a database is encrypted but MITM-able.
// Inlined rather than read from a .pem so the bundler cannot fail to trace it.
// sha256 80:70:25:AD:50:D4:ED:21:9D:2C:9C:7D:29:9C:00:4F:82:4E:B0:0C:F7:F6:5A:FE:F6:07:D0:7B:72:E6:CA:FA
const SUPABASE_CA = `-----BEGIN CERTIFICATE-----
MIIDxDCCAqygAwIBAgIUbLxMod62P2ktCiAkxnKJwtE9VPYwDQYJKoZIhvcNAQEL
BQAwazELMAkGA1UEBhMCVVMxEDAOBgNVBAgMB0RlbHdhcmUxEzARBgNVBAcMCk5l
dyBDYXN0bGUxFTATBgNVBAoMDFN1cGFiYXNlIEluYzEeMBwGA1UEAwwVU3VwYWJh
c2UgUm9vdCAyMDIxIENBMB4XDTIxMDQyODEwNTY1M1oXDTMxMDQyNjEwNTY1M1ow
azELMAkGA1UEBhMCVVMxEDAOBgNVBAgMB0RlbHdhcmUxEzARBgNVBAcMCk5ldyBD
YXN0bGUxFTATBgNVBAoMDFN1cGFiYXNlIEluYzEeMBwGA1UEAwwVU3VwYWJhc2Ug
Um9vdCAyMDIxIENBMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAqQXW
QyHOB+qR2GJobCq/CBmQ40G0oDmCC3mzVnn8sv4XNeWtE5XcEL0uVih7Jo4Dkx1Q
DmGHBH1zDfgs2qXiLb6xpw/CKQPypZW1JssOTMIfQppNQ87K75Ya0p25Y3ePS2t2
GtvHxNjUV6kjOZjEn2yWEcBdpOVCUYBVFBNMB4YBHkNRDa/+S4uywAoaTWnCJLUi
cvTlHmMw6xSQQn1UfRQHk50DMCEJ7Cy1RxrZJrkXXRP3LqQL2ijJ6F4yMfh+Gyb4
O4XajoVj/+R4GwywKYrrS8PrSNtwxr5StlQO8zIQUSMiq26wM8mgELFlS/32Uclt
NaQ1xBRizkzpZct9DwIDAQABo2AwXjALBgNVHQ8EBAMCAQYwHQYDVR0OBBYEFKjX
uXY32CztkhImng4yJNUtaUYsMB8GA1UdIwQYMBaAFKjXuXY32CztkhImng4yJNUt
aUYsMA8GA1UdEwEB/wQFMAMBAf8wDQYJKoZIhvcNAQELBQADggEBAB8spzNn+4VU
tVxbdMaX+39Z50sc7uATmus16jmmHjhIHz+l/9GlJ5KqAMOx26mPZgfzG7oneL2b
VW+WgYUkTT3XEPFWnTp2RJwQao8/tYPXWEJDc0WVQHrpmnWOFKU/d3MqBgBm5y+6
jB81TU/RG2rVerPDWP+1MMcNNy0491CTL5XQZ7JfDJJ9CCmXSdtTl4uUQnSuv/Qx
Cea13BX2ZgJc7Au30vihLhub52De4P/4gonKsNHYdbWjg7OWKwNv/zitGDVDB9Y2
CMTyZKG3XEu5Ghl1LEnI3QmEKsqaCLv12BnVjbkSeZsMnevJPs1Ye6TjjJwdik5P
o/bKiIz+Fq8=
-----END CERTIFICATE-----`;

const RATE_MAX = parseInt(process.env.WAITLIST_RATE_MAX || "5", 10);
const RATE_WINDOW = parseInt(process.env.WAITLIST_RATE_WINDOW || "60", 10) * 1000;

// Sanity gate, not full RFC 5322 — same expression as the original service.
const EMAIL = /^[^@\s]{1,64}@[^@\s]{1,255}\.[^@\s]{2,}$/;

// NO conflict target. `on conflict (email) do nothing` needs SELECT on the
// table to inspect the arbiter index, and waitlist_writer deliberately has
// none — that form fails with "permission denied for table waitlist" on every
// duplicate signup. Unqualified DO NOTHING covers the email unique constraint
// without a read. Do not "clarify" this by adding (email) back.
const INSERT = `
  insert into public.waitlist (email, source, user_agent)
  values ($1, $2, $3)
  on conflict do nothing
`;

// Module scope: warm invocations reuse the connection instead of paying a TLS
// handshake per signup. max:1 because each serverless instance handles one
// request at a time; Supavisor multiplexes across instances.
let pool = null;

function getPool() {
  if (!pool) {
    pool = new Pool({
      connectionString: process.env.WAITLIST_DATABASE_URL,
      ssl: { ca: SUPABASE_CA, rejectUnauthorized: true },
      max: 1,
      idleTimeoutMillis: 30000,
      connectionTimeoutMillis: 8000,
      statement_timeout: 8000,
      // The pool must never take the whole function down on an idle-client
      // error (Supavisor recycles connections); swallow it and let the next
      // request reconnect.
      allowExitOnIdle: true,
    });
    pool.on("error", (err) => console.error("waitlist: idle pg client error", err.message));
  }
  return pool;
}

// Alerting: a signup that fails to store is a lost customer and there is no
// second chance to capture it, so a failed insert emails an operator. Sends via
// Resend's HTTP API rather than an SMTP client, to keep `pg` the only dependency.
// No-ops silently when unconfigured, so the endpoint still works without it.
const ALERT_TO = process.env.ALERT_EMAIL_TO;
const ALERT_FROM = process.env.ALERT_EMAIL_FROM;
const RESEND_KEY = process.env.RESEND_API_KEY;
const ALERT_COOLDOWN = parseInt(process.env.ALERT_COOLDOWN_MINUTES || "15", 10) * 60000;

// Per-instance throttle. A database outage fails every request, and without
// this a traffic spike would send one email per signup. Worst case across N
// warm instances is N emails per cooldown, which is an acceptable ceiling.
let lastAlertAt = 0;

async function alertOperator(detail) {
  if (!ALERT_TO || !ALERT_FROM || !RESEND_KEY) return;

  const now = Date.now();
  if (now - lastAlertAt < ALERT_COOLDOWN) return;
  lastAlertAt = now;

  // Bounded wait: this runs before the response is sent (work after a
  // serverless response can be killed mid-flight), so it must never be the
  // reason the caller times out.
  const abort = new AbortController();
  const timer = setTimeout(() => abort.abort(), 3000);
  try {
    const r = await fetch("https://api.resend.com/emails", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${RESEND_KEY}`,
        "Content-Type": "application/json",
      },
      signal: abort.signal,
      body: JSON.stringify({
        from: ALERT_FROM,
        to: [ALERT_TO],
        subject: "[trybuddynow] waitlist signup failed",
        // Deliberately no email address in the body: the whole point of the
        // insert-only role is that signup emails are not readable outside the
        // database, and an alert inbox is a weaker place to keep them.
        text:
          `A waitlist signup could not be written to Postgres.\n\n` +
          `Error: ${detail}\n\n` +
          `Further alerts are suppressed for ${ALERT_COOLDOWN / 60000} minutes.\n` +
          `Check the Vercel function logs and Supabase project status.`,
      }),
    });
    if (!r.ok) console.error("waitlist: alert email rejected", r.status);
  } catch (err) {
    console.error("waitlist: alert email failed", err.message);
  } finally {
    clearTimeout(timer);
  }
}

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
  if (!process.env.WAITLIST_DATABASE_URL) {
    // Fail loudly in logs, quietly to the client — never leak config state.
    console.error("waitlist: WAITLIST_DATABASE_URL is unset");
    await alertOperator("WAITLIST_DATABASE_URL is unset");
    return res.status(500).end();
  }

  if (!rateOk(clientIp(req))) return res.status(429).end();

  const body = typeof req.body === "object" && req.body ? req.body : {};

  // Honeypot tripped: accept and silently drop, so the bot sees success.
  if (body.company_website) return res.status(204).end();

  const email = String(body.email || "").trim().toLowerCase();
  if (email.length > 254 || !EMAIL.test(email)) return res.status(400).end();

  try {
    await getPool().query(INSERT, [
      email,
      "landing",
      String(req.headers["user-agent"] || "").slice(0, 300),
    ]);
  } catch (err) {
    console.error("waitlist: insert failed", err.code || "", err.message);
    await alertOperator(`${err.code || "no-code"}: ${err.message}`);
    return res.status(500).end();
  }

  // 204 for both a fresh insert and a duplicate — the caller cannot tell which,
  // which is the point.
  return res.status(204).end();
};
