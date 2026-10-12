# K5 — CLI END-TO-END REPORT

**Status: NOT DONE — NOT VERIFIED.**

The `verdicts` command and the renderer exist and are unit-tested. The brief requires
the CLI to read **persisted** `EpisodeStore` and `IncidentStore` data rather than
hand-constructed objects.

That path cannot be exercised, because the pipeline does not write to `EpisodeStore`
(K4). So:

| Requirement | State |
|---|---|
| CLI reads persisted episodes | **NOT VERIFIED** |
| CLI reads persisted incidents | **NOT VERIFIED** |
| four persisted verdicts rendered | **NOT VERIFIED** |
| consequence rendering (`POSSIBLE_CONSEQUENCE_OF`) | **NOT DONE** |

No test invokes `main([...])` with `verdicts`.
