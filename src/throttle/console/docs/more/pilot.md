# Throttle Cloud

*Throttle cuts what your LLM inference costs on your own GPUs.*

The free CLI and this console show where the money goes: $/M tokens with 95%
intervals and calibrated verdicts. They stay free, local and open source (MIT).
Throttle Cloud cuts the bill.

| | |
| --- | --- |
| Start | Free, with $10 of credits, no card |
| Plans and usage rates | [throttle-pro.com/pricing](https://www.throttle-pro.com/pricing) |
| Enterprise | A hands-on pilot on your stack, priced on the savings we verify together |

## What it does

1. **Autotune** (`throttle autotune`): restarts your server with candidate
   configs from a search space you give, measures each like `throttle check`
   against a calibrated baseline, and keeps the cheapest config that is a
   verified CHEAPER and passes the OUTPUT CHANGED check. You get $/M before
   and after, the projected monthly saving and a ready-to-paste launch
   command. Try it free on a simulated server: `throttle autotune --dry-run`.
2. **CI cost gate** (`throttle gate`): fails a deploy that is MORE EXPENSIVE
   or OUTPUT CHANGED.
3. **Team history** (`throttle check --push`): every check in one place, with
   a link to share.

OUTPUT CHANGED compares answer length and how often answers hit the token cap;
it does not check that answers are correct.

## What leaves your machine

Only when you are logged in and run a cloud command: numbers, the model name,
the config fingerprint (secret-looking values masked), engine and GPU. Never
prompts, responses, endpoint URLs, hostnames or keys.

## Get started

- `throttle login` (paste an API key from the dashboard)
- `throttle upgrade` prints the plans and a QR code for the pricing page.
- Enterprise: email **kush@throttle-pro.com** with the subject "Throttle pilot".
