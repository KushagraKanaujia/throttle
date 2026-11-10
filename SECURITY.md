# Security Policy

## Supported Versions

We actively support the latest released version of Throttle with security updates.

| Version | Supported          |
| ------- | ------------------ |
| 0.7.x   | :white_check_mark: |
| < 0.7   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in Throttle, please report it privately to help us fix it before public disclosure.

**Please do NOT open a public GitHub issue for security vulnerabilities.**

### How to Report

Email security reports to: **kush@throttle-pro.com**

Include in your report:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if you have one)

### What to Expect

- **Acknowledgment**: We'll acknowledge your report within 48 hours
- **Updates**: We'll keep you informed of our progress
- **Fix timeline**: We aim to release fixes for critical vulnerabilities within 7 days
- **Credit**: We'll credit you in the security advisory (unless you prefer to remain anonymous)

## Security model and data flow

**What runs where.**

- **A proxy you run.** `throttle capture` binds to `127.0.0.1` by default and warns you if you bind it anywhere else.
- **Your CI.** The GitHub Action runs on your runner with your workflow token. There is no Throttle server.
- **The CLI.** Every other command runs where you run it. Data stays in `~/.throttle/own` (or `$THROTTLE_OWN_DIR`).

**What leaves your infra.**

- The API calls your app already makes, unchanged.
- Judge calls go to the provider and key you choose.
- Candidate API models see the prompts you send them.
- Throttle the company receives nothing. There is no telemetry.

**Keys** are read from environment variables you name. They are never printed or written to disk, and capture never stores auth headers.

**Redaction** of captured bodies: `mask` (default), `hash`, `metadata` (no bodies) or `off`. Pattern redaction does not catch names, street addresses or free-text details; use `metadata` if that matters.

**Supply chain.** Releases go to PyPI through trusted publishing (OIDC), so no upload token is stored, and the release workflow pins every action by SHA. Pin your install and the GitHub Action to a full commit SHA. The capture proxy sees your API keys in flight, so run only code you have pinned.

### Security Best Practices for Users

1. **Never commit credentials**: Throttle sanitizes reports, but operators must never include credentials in CLI arguments or metadata
2. **Use HTTPS endpoints**: the serving tools (`plan`, `smoke`, `benchmark`) refuse plain HTTP to a non-loopback host unless you pass `--allow-insecure-http`
3. **Pin by SHA**: install Throttle and its GitHub Action from a full commit SHA, not a branch or tag
4. **Keep the proxy local**: leave `throttle capture` on `127.0.0.1` unless something in front of it controls who can reach it
5. **Keep dependencies updated**: Run `pip list --outdated` regularly

## Scope

Security issues in scope:
- Command injection vulnerabilities
- Credential leakage in saved reports
- Unsafe HTTP request handling
- Privilege escalation
- Dependency vulnerabilities

Out of scope:
- Issues in third-party inference servers (report to vLLM, Ollama, etc.)
- Denial of service via excessive load sent on purpose through your own proxy or benchmark
- Social engineering attacks

## Disclosure Policy

We follow coordinated disclosure:
1. You report the vulnerability privately
2. We confirm and develop a fix
3. We release the patched version
4. We publish a security advisory with credit
5. Public disclosure happens after the fix is released

Thank you for helping keep Throttle and its users secure!
