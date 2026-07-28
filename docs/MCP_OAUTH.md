# Connecting Claude to Snoocle's MCP server

Add `https://<your-service>/mcp` as a custom connector in Claude. You'll be
asked for your `SNOOCLE_API_TOKEN` once on a consent screen, and that's it.

## Why this exists

Snoocle originally gated `/mcp` with a static bearer token. Claude's remote-MCP
connector can't use one. It expects an OAuth 2.1 flow: discover an
authorization server, register itself, get a user to consent, exchange a code
for a token. When it couldn't find an authorization server to register with, it
failed with:

> Couldn't register with Snoocle MCP's sign-in service.

That message points at dynamic client registration, but the real gap was
earlier — there was no OAuth anything to register *with*.

Snoocle now acts as its own authorization server. For a single-user personal
service that's the simplest correct arrangement: no second deployment, no
third-party identity provider, and the token you already have is what proves
you're the owner.

## What happens when you connect

1. Claude requests `/mcp` with no token and gets a **401** carrying
   `WWW-Authenticate: Bearer resource_metadata="…"`. Claude only reads that
   header on a 401 — on a 200 it's ignored — so the status code is part of the
   protocol, not an accident.
2. It fetches `/.well-known/oauth-protected-resource`, which names this server
   as the resource and points at the authorization server (RFC 9728).
3. It fetches `/.well-known/oauth-authorization-server` for the endpoints
   (RFC 8414).
4. It registers itself at `/oauth/register` (RFC 7591) and gets a `client_id`.
5. It opens `/oauth/authorize` in your browser. **You enter your
   `SNOOCLE_API_TOKEN`.** That's the only manual step.
6. You're redirected back with a code, which Claude exchanges at `/oauth/token`
   for an access token and a refresh token.

Claude refreshes on its own after that. Access tokens last an hour; refresh
tokens last 90 days and rotate on every use.

## What stayed the same

The REST API is untouched. The iOS app and the admin UI still send
`Authorization: Bearer $SNOOCLE_API_TOKEN` and nothing about them changed.

OAuth tokens are minted with `/mcp` as their audience and are accepted **only**
there. Presenting one to `/v1/songs` returns 401 — honouring it would be exactly
the audience confusion the MCP spec forbids.

## Security notes

Worth knowing what is and isn't defended:

- **PKCE is mandatory and S256-only.** `plain` is refused rather than accepted
  as a downgrade, so intercepting an authorization code doesn't get anyone a
  token.
- **Redirect URIs match exactly.** No prefixes, no wildcards. The one exception
  is loopback (`http://localhost/callback`), where the port is ignored because
  Claude Code binds an ephemeral one — RFC 8252 §7.3 requires this.
- **Errors before the redirect URI is verified render a page instead of
  redirecting.** Bouncing to an unverified URI is the open redirect the check
  exists to prevent.
- **Authorization codes are single-use and live 60 seconds.** A replayed code
  is burned whether or not it was still valid.
- **Refresh tokens rotate.** The old one is invalidated in the same transaction
  that issues the new one, as OAuth 2.1 requires for public clients.
- **Everything is stored, not held in memory.** Cloud Run scales to zero; an
  in-memory client registry would silently unauthorize the connector on every
  cold start, which reads to a user as "it keeps asking me to reconnect".

The consent screen shows the redirect destination, and warns explicitly when it
is a loopback address — any local process can bind a port and claim to be a
legitimate client, and a Client ID Metadata Document can't prevent that on its
own.

## If it doesn't connect

**"Couldn't reach the MCP server."** Discovery failed. Check that
`/.well-known/oauth-protected-resource` is reachable from the public internet
and that the `resource` field matches the URL you typed into Claude *exactly*,
path included.

**Metadata advertises the wrong host.** The service builds every URL from the
forwarded headers. If something in front of it rewrites those, set
`SNOOCLE_PUBLIC_URL` explicitly.

**It connects, then asks again later.** Check the store backend is Firestore,
not memory — the in-memory one is per-process and disappears on a cold start.

Useful check without Claude in the loop:

```sh
curl -i https://<service>/mcp -X POST -d '{}'          # expect 401 + header
curl -s https://<service>/.well-known/oauth-protected-resource | jq
curl -s https://<service>/.well-known/oauth-authorization-server | jq
```
