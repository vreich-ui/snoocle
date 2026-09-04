# snoocle

- What it is: the Python MIR / agent server behind the Snoocle song-practice
  app. `snoocle_server/` is the service, `studio/` is the JS workbench,
  `tests/` + `tests_js/` are the two test planes.
- GitHub `vreich-ui/snoocle`. No Netlify site. The iOS app lives separately in
  `~/xCode/Snoocle` (`vreich-ui/Snoocle-iOS`).
- Tests: `python3 -m pytest` for the server; the `studio` job covers the JS.
  CI gates on `python` and `studio`.
- Land with `/ship snoocle <branch>`. `main` is protected.
- Never touch: deterministic pipeline outputs. Song timing changes must go
  through the deterministic tools, not hand-edited JSON.
