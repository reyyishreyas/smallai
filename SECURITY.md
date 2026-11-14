# Security Policy

## Scope

SmallAI is an **offline, CPU-only** library. It performs no network access,
downloads no models, and executes no user-supplied code. The realistic security
surface is therefore small: untrusted image files passed to Pillow, and any
service built on top of the gate.

## Reporting a vulnerability

Please **do not** open a public issue for a security problem.

Report it privately via GitHub's
[private vulnerability reporting](https://github.com/reyyishreyas/smallai/security/advisories/new)
or by email to the maintainer listed in `pyproject.toml`.

Include:

- affected version or commit,
- a minimal reproduction (image or path that triggers it),
- impact and any suggested mitigation.

You can expect an acknowledgement within 72 hours.

## Supported versions

Only the latest release on `main` receives fixes.

| Version | Supported |
| --- | --- |
| 0.1.x | Yes |
| < 0.1 | No |

## Notes for deployers

- Treat all image input as untrusted; run batch jobs with a memory cap.
- The gate returns a quality verdict, not a security decision — do not use it as
  an authorization or content-safety control.
- Datasets and corpora are not distributed with this repository.
