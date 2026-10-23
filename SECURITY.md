# Security and public releases

Keep real credentials in a local `.env`, starting from `.env.example`. Never
commit credentials, browser profiles, personal editor settings, or raw run logs.
The credentials intentionally shown in the local webhack sandbox are fictional
test fixtures, not credentials for real systems.

This repository was created from a checked file snapshot with a new initial
commit. It does not share the previous repository's history or pull-request
references. Never push or merge old Git history into this repository: doing so
could reintroduce credentials or browser profiles. Clone this repository for
future work.

Removing Git history does not revoke a credential. Owners should separately
review the security of credentials retained in other private repositories and
rotate any that have been exposed outside their intended access boundary.

GitHub pull-request references and cached views can retain old commits after a
force push. Repository owners should follow GitHub's
[sensitive-data removal procedure](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)
and contact GitHub Support if those references require removal. A clean branch
scan alone does not establish that all server-side copies have been purged.

Model launchers, proxies, and sandbox servers bind to loopback by default.
Use `HOST` on shell launchers or `--host` on proxies only when remote access is
required and separately protected. The example local API strings are not an
access-control boundary for a remotely exposed service.

Before pushing, install [Gitleaks](https://github.com/gitleaks/gitleaks) and run:

```sh
gitleaks dir . --redact
gitleaks git . --log-opts="--all" --redact
```

The configuration permits only specific, known fictional credential strings at
their fixture locations. Do not broaden those exceptions to hide a real key.
Check all branches and tags before a public release. Separately confirm dataset
provenance, redistribution permissions, and that any person-like fixture data
is synthetic; credential scanning cannot establish those facts.
