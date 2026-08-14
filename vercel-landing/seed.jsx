/* =========================================================
   seed — believable buddy state
   ========================================================= */

const SEED = {
  user: "nate",
  backend: "ark.mit.edu",

  approvals: [
    {
      id: "ap1",
      src: "mail.google.com",
      when: "4m ago",
      tag: "draft reply",
      title: "Reply to Priya re: Q3 roadmap review",
      body: "She asked to move the review to Thursday and wants the metrics deck attached. I drafted a yes with the deck linked.",
      plan: [
        "send reply confirming Thursday 2pm",
        "attach roadmap-metrics.pdf from drive",
        "add the slot to your calendar",
      ],
      tools: ["gmail.send", "drive.read", "calendar.write"],
    },
    {
      id: "ap2",
      src: "linear.app",
      when: "22m ago",
      tag: "triage",
      title: "Close 6 stale issues in team ark",
      body: "Six issues have had no activity in 30+ days and are unassigned. I'd comment, label them stale, and close.",
      plan: [
        "comment \"closing as stale — reopen if still relevant\"",
        "apply label: stale",
        "move to Canceled",
      ],
      tools: ["linear.write"],
    },
    {
      id: "ap3",
      src: "calendar.google.com",
      when: "1h ago",
      tag: "scheduling",
      title: "Decline the overlapping 'sync' invite",
      body: "This 3pm sync overlaps your focus block. You've skipped the last two. I'd decline with a note offering async.",
      plan: [
        "decline 'weekly sync' for fri 3pm",
        "reply: \"on a focus block — happy to read notes async\"",
      ],
      tools: ["calendar.write", "gmail.send"],
    },
  ],

  tasks: [
    {
      id: "t1", state: "run", when: "running 2m",
      text: "drafting the Q3 metrics summary", src: "from drive + linear",
      events: [
        { k: "read", t: "opened roadmap-metrics.pdf" },
        { k: "linear", t: "pulled 14 completed issues this cycle" },
        { k: "think", t: "grouping by theme: velocity, quality, scope" },
        { k: "write", t: "draft 1 — 380 words", ok: true },
      ],
    },
    {
      id: "t2", state: "run", when: "running 8m",
      text: "watching inbox for the AWS invoice", src: "mail.google.com",
      events: [
        { k: "watch", t: "filter: from billing@amazon, subject ~ invoice" },
        { k: "idle", t: "no match yet — checked 12m ago" },
      ],
    },
    {
      id: "t3", state: "done", when: "done 18m ago",
      text: "unsubscribed from 9 marketing senders", src: "mail.google.com",
      events: [
        { k: "scan", t: "found 9 senders, 0 opens in 90 days" },
        { k: "act", t: "clicked unsubscribe + filtered to trash", ok: true },
      ],
    },
  ],

  watching: [
    { id: "w1", src: "linear.app / team ark", note: "new issues + status changes", cadence: "every 5m", live: true },
    { id: "w2", src: "mail.google.com / inbox", note: "anything that needs a reply", cadence: "live", live: true },
    { id: "w3", src: "calendar.google.com", note: "conflicts + prep needed", cadence: "every 15m", live: true },
    { id: "w4", src: "github.com / arkos", note: "review requests on your PRs", cadence: "every 10m", live: false },
  ],

  files: {
    "/home/nate": [
      { name: "drafts", dir: true },
      { name: "downloads", dir: true },
      { name: "notes.md", size: "2.1 kb" },
      { name: "roadmap-metrics.pdf", size: "412 kb" },
      { name: ".buddyrc", size: "640 b" },
    ],
    "/home/nate/drafts": [
      { name: "..", dir: true, up: true },
      { name: "q3-summary.md", size: "1.8 kb" },
      { name: "reply-priya.txt", size: "320 b" },
    ],
    "/home/nate/downloads": [
      { name: "..", dir: true, up: true },
      { name: "invoice-may.pdf", size: "88 kb" },
    ],
  },

  fileBodies: {
    "/home/nate/notes.md":
`# scratch

- ask buddy to keep the linear board tidy on fridays
- the AWS invoice usually lands ~3rd of the month
- priya prefers async; default to notes over meetings

## standing rules for buddy
1. never send money or sign anything without asking
2. triage freely, but show me the plan before closing issues
3. quiet hours 9pm–7am — hold non-urgent pings`,

    "/home/nate/drafts/q3-summary.md":
`# Q3 — what shipped

velocity is up ~18% over Q2. three themes:

**velocity** — cut cycle time from 6.2d to 4.9d median.
**quality** — escaped-bug rate down by a third.
**scope** — shipped the agent runner a cycle early.

(draft — buddy, tighten the quality section)`,

    "/home/nate/drafts/reply-priya.txt":
`Hi Priya — Thursday 2pm works. I've attached the
roadmap metrics deck. See you then.`,

    "/home/nate/.buddyrc":
`backend = ark.mit.edu
user    = nate
quiet   = 21:00-07:00
autonomy = triage:high  send:ask  spend:never`,
  },

  computerTasks: [
    { id: "c1", state: "run", text: "indexing ~/drafts for the summary", when: "running" },
    { id: "c2", state: "done", text: "compressed downloads/ — freed 1.2 gb", when: "12m ago" },
  ],

  chat: [
    { who: "you", text: "keep an eye on the linear board and tidy it up on fridays" },
    { who: "buddy", text: "done. i'll sweep team ark every friday at 4pm — comment + close anything stale, and surface the list here before i close so you can veto. sound right?" },
    { who: "you", text: "yep. and never close anything assigned to a person" },
    { who: "buddy", text: "noted as a standing rule — assigned issues are off-limits. only unassigned + 30d quiet get triaged." },
  ],
};

const REPLIES = [
  "on it. i'll workshop a plan and drop it in approvals before doing anything.",
  "noted. i'll watch for that quietly and surface it when it matters.",
  "got it — give me a sec to look across your inbox and calendar.",
  "thinking out loud with you. nothing leaves the desk without your ok.",
];

window.SEED = SEED;
window.REPLIES = REPLIES;
