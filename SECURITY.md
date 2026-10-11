# Security Policy

## Supported Versions

We actively support the latest released version of Throttle with security updates.

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | :white_check_mark: |
| 0.2.x   | :x:                |
| 0.1.x   | :x:                |

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

### Security Best Practices for Users

Throttle is designed with security in mind:

1. **Never commit credentials**: Throttle sanitizes reports, but operators must never include credentials in CLI arguments or metadata
2. **Use HTTPS endpoints**: Plain HTTP is only allowed for localhost/loopback
3. **Review validation artifacts**: All JSON outputs are sanitized and safe to share
4. **Keep dependencies updated**: Run `pip list --outdated` regularly

## Scope

Security issues in scope:
- Command injection vulnerabilities
- Credential leakage in saved reports
- Unsafe HTTP request handling
- Privilege escalation
- Dependency vulnerabilities

Out of scope:
- Issues in third-party inference servers (report to vLLM, Ollama, etc.)
- Denial of service via excessive load (this is a benchmarking tool)
- Social engineering attacks

## Disclosure Policy

We follow coordinated disclosure:
1. You report the vulnerability privately
2. We confirm and develop a fix
3. We release the patched version
4. We publish a security advisory with credit
5. Public disclosure happens after the fix is released

Thank you for helping keep Throttle and its users secure!
