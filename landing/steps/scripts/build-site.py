#!/usr/bin/env python3
"""
Build the deployable landing page from the source Landing.html by applying two
changes, so we never hand-maintain a second copy:

  1. Add a hidden honeypot input to each signup form.
  2. Replace the inline (no-op) handleSignup with one that POSTs the email to
     /landing/api/waitlist and always shows success (no enumeration).

Usage: build-site.py <src Landing.html> <out index.html>
"""
import re
import sys

HONEYPOT = (
    '<input type="text" name="company_website" tabindex="-1" autocomplete="off" '
    'aria-hidden="true" style="position:absolute;left:-9999px;width:1px;height:1px;opacity:0">'
)

NEW_HANDLER = """function handleSignup(e){
e.preventDefault();
const form=e.target;
const hp=form.querySelector('input[name="company_website"]');
const emailEl=form.querySelector('input[type=email]');
const email=(emailEl?emailEl.value:'').trim().toLowerCase();
// Bots that fill the honeypot: pretend success, send nothing.
if(!(hp&&hp.value)){
  fetch('/landing/api/waitlist',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({email:email})
  }).catch(function(){});
}
form.classList.add('done'); // always show success (no enumeration, no leak)
return false;
}"""


def main() -> None:
    src, out = sys.argv[1], sys.argv[2]
    html = open(src, encoding="utf-8").read()

    # 1. honeypot before each </form> of the signup forms
    html = re.sub(
        r'(<form class="signup"[^>]*>)(.*?)(</form>)',
        lambda m: m.group(1) + m.group(2) + HONEYPOT + m.group(3),
        html,
        flags=re.DOTALL,
    )

    # 2. swap the handler
    html = re.sub(
        r"function handleSignup\(e\)\{.*?\n\}",
        NEW_HANDLER,
        html,
        count=1,
        flags=re.DOTALL,
    )

    open(out, "w", encoding="utf-8").write(html)
    print("wrote", out)


if __name__ == "__main__":
    main()
