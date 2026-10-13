# ApexRL 1.0

## Play on Windows

Extract the **entire** `ApexRL-1.0.0-Windows-x64.zip` into a writable folder, then double-click **ApexRL.exe**. Keep its `_internal` folder beside it. The Windows package includes Python, game dependencies, CPU AI playback, and both trained drivers; no Python installation or download is needed to play.

Choose **Play** or **Track Selection** for manual time trials. Choose **AI Garage → Watch Driver** for AI racing. The garage lists the two published drivers first; **Show Training Runs** includes development checkpoints if you add them to a source checkout.

- **All Circuits V1:** reliable single laps on Rookie Ring, Apex Valley, and Velocity Circuit. Prior validation completed 25/25 fresh-grid laps per circuit.
- **Rookie Reliable V1:** a faster Rookie Ring specialist. Its validation does not establish competence on the harder circuits.

The selected circuit shows measured valid-lap success and best validated time. Loading stays responsive and can be cancelled. AI laps never update human personal bests.

## Controls and saves

**W/Up** accelerates; **S/Down** brakes; **A/Left and D/Right** steer. **Escape** pauses, **R** recovers the manual car or restarts an AI episode, **Tab** opens the circuit overview, and **F3** toggles telemetry. Recovery invalidates a manual lap. Menus support arrows/Enter and mouse input.

Settings and human lap records live in **saves/** beside the executable. Keep that folder when replacing the game. Runtime diagnostics are written to `saves/diagnostics.log`. Low graphics detail and windowed/fullscreen controls are available under Settings.

## Verify or develop

The source archive includes configuration, tests, training tools, both published checkpoints, and their reports. It requires Python 3.11+; follow README setup instructions. Manual play keeps RL packages optional. To rebuild Windows, install `requirements-rl.txt` and `requirements-build.txt`, then run `python build_windows.py`.

The packaged executable can verify manual acceleration/pause and every validated driver/circuit pairing:

```powershell
.\ApexRL.exe --self-check --output verification/report.json
```

This uses a headless SDL display and writes measured lap outcomes to JSON. Native screenshots and release verification are recorded in `RELEASE_VALIDATION.md` in the source checkout. Third-party runtime notices are included under **licenses/**. Archives have SHA-256 checksums in `SHA256SUMS.txt`.

This release provides original single-player time trials and trained AI spectator racing. It does not add multiplayer, opponents, gamepad support, or a professional tire model. The all-circuit driver prioritizes reliable laps over aggressive lap times.
