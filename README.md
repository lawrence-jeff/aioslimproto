aioslimproto
==================================

> **This branch (`ma-squeezelite-browse`)** carries a working set of patches on top of upstream
> `main`, developed and tested against Music Assistant's Squeezelite provider to let a client
> (e.g. JiveLite/piCorePlayer, or a real Squeezebox) browse the MA library and stay in sync with
> queue/playback changes, instead of only playing whatever MA pushes to it. It's a fork branch for
> other MA/aioslimproto developers to read and build on - not an upstream release, and not meant
> to run standalone (the matching MA-side provider patches live on the
> [`ma-squeezelite-browse` branch of lawrence-jeff/server](https://github.com/lawrence-jeff/server/tree/ma-squeezelite-browse),
> which pins its `aioslimproto` requirement to this branch). See
> [lawrence-jeff/MA-SqueezeliteBrowse](https://github.com/lawrence-jeff/MA-SqueezeliteBrowse) for
> the full project, including the browse-specific provider code and the `reinject.sh` workflow
> used to test against a live container. Individual pieces of this are also up as separate PRs
> against upstream - see that repo's README for current status.
>
> Everything below this point is upstream's own README, unmodified.

[![pypi_badge](https://img.shields.io/pypi/v/aioslimproto.svg)](https://pypi.python.org/pypi/aioslimproto)

**AIOSlimProto**


SLIMProto implementation in async python allows you to control squeezebox players (and compatibles).

Requires Python 3.11+ and uses asyncio.

For simple usage examples, see the example script in the scripts folder.


For a full reference implementation, see [Home Assistant](https://github.com/home-assistant/core/tree/dev/homeassistant/components/slimproto) and [Music Assistant](https://github.com/music-assistant/server/tree/main/music_assistant/providers/squeezelite)
