"""Site-specific discovery sources (as opposed to `discovery.search`'s
generic web search + `discovery.chordsheet`'s generic parser).

Personal-use tool: a site-specific scraper is acceptable here (see the
master plan's guardrails), but stays isolated in its own module, behind its
own config flag, so it can be disabled instantly and never takes the whole
discovery pipeline down with it.
"""
