"""
Basic implementation of CLI control for SlimProto players.

Some players (e.g. PiCorePlayer) use the jsonrpc api to control for example volume remotely.

Both the (legacy) Telnet CLI and the JSON RPC interface are supported.
This is a very basic implementation that only fulfills commands needed by those players,
other commands will be published as-is on the eventbus for library consumers to act on.
there's no support for media browsing through this minimal api, this is NOT a replacement for
the Logitech Media Server.

https://github.com/elParaguayo/LMS-CLI-Documentation/blob/master/LMS-CLI.md
https://gist.github.com/samtherussell/335bf9ba75363bd167d2470b8689d9f2
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import contextlib
from contextlib import suppress
from dataclasses import dataclass, field
import inspect
import json
import time
from typing import TYPE_CHECKING, Any
import urllib.parse
from uuid import uuid1

from aiohttp import web

from aioslimproto.client import PlayerState
from aioslimproto.util import empty_queue, select_free_port

from .models import (
    PLAYMODE_MAP,
    CometDResponse,
    CommandErrorMessage,
    CommandMessage,
    CommandResultMessage,
    EventType,
    MediaDetails,
    MediaMetadata,
    PlayerItem,
    PlayersResponse,
    PlayerStatusResponse,
    PlaylistItem,
    ServerStatusResponse,
    SlimEvent,
    SlimMenuItem,
    SlimSubscribeMessage,
)

if TYPE_CHECKING:
    from .client import SlimClient
    from .server import SlimServer

# ruff: noqa: ARG002, FBT001, FBT002, RUF006

SlimCLIScalar = str | int | float

ArgsType = list[SlimCLIScalar]
"""Known as positional parameters in CLI docs."""
KwargsType = dict[str, SlimCLIScalar]
"""Known as tagged parameters in CLI docs."""

SlimCLICommandResponse = (
    None | SlimCLIScalar | list[SlimCLIScalar | dict[str, SlimCLIScalar]]
)
"""
Command handlers can return this value, and `aioslimproto` will take care of serialization.

- None: command handled but no response, the CLI will echo the request back to the client.
- list of scalars of the same length as the number of `?` in the query: the `?` will be replaced.
- otherwise, the list is just appended to the request; dicts represent ordered blocks of output.
"""


@dataclass
class SlimCLICommand:
    """Representation of a CLI command with minimal parsing."""

    player_id: str | None
    command: str
    args: list[SlimCLIScalar]
    kwargs: dict[str, SlimCLIScalar]


SlimCLICommandHandler = Callable[
    [SlimCLICommand], SlimCLICommandResponse | Awaitable[SlimCLICommandResponse]
]


@dataclass
class CometDClient:
    """Representation of a connected CometD client."""

    client_id: str
    player_id: str = ""
    queue: asyncio.Queue[CometDResponse] = field(
        default_factory=lambda: asyncio.Queue(maxsize=50)
    )
    last_seen: int = int(time.time())
    first_event: CometDResponse | None = None
    meta_subscriptions: set[str] = field(default_factory=set)
    slim_subscriptions: dict[str, SlimSubscribeMessage] = field(default_factory=dict)
    streaming: bool = False


def parse_value(
    raw_value: SlimCLIScalar,
) -> SlimCLIScalar | tuple[str, SlimCLIScalar]:
    """
    Transform API param into a usable value.

    Integer/float values are sometimes sent as string so we try to parse that.
    """
    if isinstance(raw_value, str):
        key = None
        value = raw_value
        if ":" in raw_value:
            # this is a key:value pair
            key, value = raw_value.split(":", 1)

        with contextlib.suppress(ValueError):
            # number that starts with a + indicates a relative value
            if not value.startswith("+"):
                value = float(value)
                if value.is_integer():
                    value = int(value)

        if key is not None:
            return (key, value)

        return value
    return raw_value


def parse_args(raw_values: list[SlimCLIScalar]) -> tuple[ArgsType, KwargsType]:
    """Parse positional and tagged parameters from raw CLI tokens."""
    args: ArgsType = []
    kwargs: KwargsType = {}
    for raw_value in raw_values:
        value = parse_value(raw_value)
        if isinstance(value, tuple):
            kwargs[value[0]] = value[1]
        else:
            args.append(value)
    return (args, kwargs)


class SlimProtoCLI:
    """Basic implementation of CLI control for SlimProto players."""

    _unsub_callback: Callable | None = None
    _periodic_task: asyncio.Task | None = None
    _cli_server: asyncio.Server | None = None
    command_handler: SlimCLICommandHandler | None = None
    # Set by the provider after construction. Returns MA's display_name for a
    # player_id, or None if MA doesn't know that player. Independent of
    # command_handler so the name works without the browse feature.
    display_name_lookup: Callable[[str], str | None] | None = None
    # Set by the provider after construction. The Music Assistant instance used
    # for seek and play. Independent of command_handler so these work without browse.
    mass: Any | None = None

    def __init__(
        self,
        server: SlimServer,
        cli_port: int | None = None,
        cli_port_json: int | None = 0,
        command_handler: SlimCLICommandHandler | None = None,
        extra_routes: dict[str, Callable] | None = None,
    ) -> None:
        """
        Initialize Telnet and/or Json interface CLI.

        Set port to None to disable the interface, set to 0 for auto select a free port.
        """
        self.server = server
        self.cli_port = cli_port
        self.cli_port_json = cli_port_json
        self.logger = server.logger.getChild("cli")
        self.command_handler = command_handler
        self.extra_routes = extra_routes or {}
        self._cometd_clients: dict[str, CometDClient] = {}
        self._player_map: dict[str, str] = {}
        self._apprunner: web.AppRunner | None = None
        self._webapp: web.Application | None = None
        self._tcp_site: web.TCPSite | None = None

    async def start(self) -> None:
        """Start running the server(s)."""
        # if port is specified as 0, auto select a free port for the cli/json interface
        if self.cli_port == 0:
            self.cli_port = await select_free_port(9090, 9190)
        if self.cli_port_json == 0:
            self.cli_port_json = await select_free_port(9000, 9089)
        if self.cli_port is not None:
            self.logger.info(
                "Starting (legacy/telnet) SLIMProto CLI on port %s",
                self.cli_port,
            )
            self._cli_server = await asyncio.start_server(
                self._handle_cli_client,
                "0.0.0.0",  # noqa: S104
                self.cli_port,
            )
        if self.cli_port_json is not None:
            self.logger.info(
                "Starting SLIMProto JSON RPC CLI on port %s",
                self.cli_port_json,
            )
            self._webapp = web.Application(
                logger=self.logger,
            )
            self._apprunner = web.AppRunner(self._webapp, access_log=None)
            self._webapp.router.add_route(
                "*",
                "/jsonrpc.js",
                self._handle_jsonrpc_client,
            )
            self._webapp.router.add_route("*", "/cometd", self._handle_cometd_client)
            for path, handler in self.extra_routes.items():
                self._webapp.router.add_route("*", path, handler)
            await self._apprunner.setup()
            # set host to None to bind to all addresses on both IPv4 and IPv6
            self._tcp_site = web.TCPSite(
                self._apprunner,
                host=None,
                port=self.cli_port_json,
                shutdown_timeout=10,
            )
            await self._tcp_site.start()
        # setup subscriptions
        self._unsub_callback = self.server.subscribe(
            self._on_player_event,
            (
                EventType.PLAYER_UPDATED,
                EventType.PLAYER_CONNECTED,
                EventType.PLAYER_PRESETS_UPDATED,
            ),
        )
        self._periodic_task = asyncio.create_task(self._do_periodic())

    async def stop(self) -> None:
        """Stop running the server(s)."""
        # stop/clean json-rpc webserver
        if self._tcp_site:
            await self._tcp_site.stop()
            self._tcp_site = None
        if self._apprunner:
            await self._apprunner.cleanup()
            self._apprunner = None
        if self._webapp:
            await self._webapp.shutdown()
            await self._webapp.cleanup()
            self._webapp = None
        # stop cli server
        if self._cli_server:
            self._cli_server.close()
            self._cli_server = None
        # cleanup callbacks and tasks
        if self._unsub_callback:
            self._unsub_callback()
            self._unsub_callback = None
        if self._periodic_task:
            self._periodic_task.cancel()
            self._periodic_task = None

    async def _handle_cli_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle new connection on the legacy CLI."""
        # https://raw.githubusercontent.com/Logitech/slimserver/public/7.8/HTML/EN/html/docs/cli-api.html
        # https://github.com/elParaguayo/LMS-CLI-Documentation/blob/master/LMS-CLI.md
        self.logger.debug("Client connected on Telnet CLI")
        try:
            while True:
                raw_request = await reader.readline()
                raw_request = raw_request.strip().decode("utf-8")
                if not raw_request:
                    break
                # request comes in as url encoded strings, separated by space
                raw_params = [urllib.parse.unquote(x) for x in raw_request.split(" ")]
                # the first param is either a macaddress or a command
                if ":" in raw_params[0]:
                    # assume this is a mac address (=player_id)
                    player_id = raw_params[0]
                    command = raw_params[1]
                    command_params = raw_params[2:]
                else:
                    player_id = ""
                    command = raw_params[0]
                    command_params = raw_params[1:]

                args, kwargs = parse_args(command_params)
                slim_command = SlimCLICommand(
                    player_id=player_id or None,
                    command=command,
                    args=args,
                    kwargs=kwargs,
                )
                self.logger.debug(
                    "Handling CLI-request (player: %s command: %s - args: %s - kwargs: %s)",
                    player_id,
                    command,
                    str(args),
                    str(kwargs),
                )
                try:
                    cmd_result = await self._dispatch_command(slim_command)
                    response = self._format_cli_response(raw_params, cmd_result)
                except NotImplementedError as err:
                    # No handler found, forward as event and echo the request.
                    self.logger.debug("No handler found", exc_info=err)
                    self._publish_unhandled_command(slim_command)
                    response = " ".join(urllib.parse.quote(x) for x in raw_params)
                response += "\n"
                writer.write(response.encode("iso-8859-1"))
                await writer.drain()
        except ConnectionResetError:
            pass
        except Exception as err:  # noqa: BLE001
            self.logger.debug("Error handling CLI command", exc_info=err)
        finally:
            self.logger.debug("Client disconnected from Telnet CLI")
            # Properly close the writer to release socket resources
            if not writer.is_closing():
                writer.close()
                with suppress(Exception):
                    await writer.wait_closed()

    async def _dispatch_command(self, slim_command: SlimCLICommand) -> Any:
        """Dispatch a command to the application handler, then the built-in fallback."""
        if self.command_handler is not None:
            try:
                result = self.command_handler(slim_command)
                if inspect.isawaitable(result):
                    result = await result
            except NotImplementedError:
                pass
            else:
                return result

        handler = getattr(self, f"_handle_{slim_command.command}", None)
        if handler is None:
            raise NotImplementedError(f"No handler for command: {slim_command.command}")
        result = handler(
            slim_command.player_id or "",
            *slim_command.args,
            **slim_command.kwargs,
        )
        if inspect.isawaitable(result):
            result = await result
        return result

    def _publish_unhandled_command(self, slim_command: SlimCLICommand) -> None:
        """Publish an unhandled command to the addressed player, when available."""
        if player := self.server.get_player(slim_command.player_id or ""):
            args_str = " ".join(
                [slim_command.command] + [str(arg) for arg in slim_command.args]
            )
            player.callback(player, EventType.PLAYER_CLI_EVENT, args_str)
        else:
            self.logger.warning(
                "No handler for %s (player: %s - args: %s - kwargs: %s)",
                slim_command.command,
                slim_command.player_id,
                str(slim_command.args),
                str(slim_command.kwargs),
            )

    def _format_cli_response(self, raw_params: list[str], cmd_result: Any) -> str:
        """Format a command result for the Telnet CLI transport."""
        if cmd_result is None:
            return " ".join(urllib.parse.quote(param) for param in raw_params)

        if isinstance(cmd_result, (int, str, float)):
            cmd_result = [cmd_result]

        if isinstance(cmd_result, list):
            question_indexes = [
                index for index, value in enumerate(raw_params) if value == "?"
            ]
            if question_indexes and len(cmd_result) == len(question_indexes):
                if any(isinstance(value, dict) for value in cmd_result):
                    raise ValueError
                for index, value in zip(question_indexes, cmd_result, strict=True):
                    raw_params[index] = str(value)
                return " ".join(urllib.parse.quote(param) for param in raw_params)

            result_parts: list[str] = []
            for value in cmd_result:
                if isinstance(value, dict):
                    result_parts.extend(dict_to_strings(value))
                else:
                    result_parts.append(str(value))
            response = " ".join(urllib.parse.quote(param) for param in raw_params)
            if result_parts:
                response += " " + " ".join(
                    urllib.parse.quote(value) for value in result_parts
                )
            return response

        # For backwards compatibility, to be removed whenever possible.
        if isinstance(cmd_result, dict):
            result_parts = dict_to_strings(cmd_result)
            response = " ".join(urllib.parse.quote(param) for param in raw_params)
            if result_parts:
                response += " " + " ".join(
                    urllib.parse.quote(value) for value in result_parts
                )
            return response
        return " ".join(
            urllib.parse.quote(param) for param in [*raw_params, str(cmd_result)]
        )

    async def _handle_jsonrpc_client(self, request: web.Request) -> web.Response:
        """Handle request on JSON-RPC endpoint."""
        command_msg: CommandMessage = await request.json()
        self.logger.debug("Received request: %s", command_msg)
        cmd_result = await self._handle_command(command_msg["params"])
        if cmd_result is None:
            result: CommandErrorMessage = {
                **command_msg,
                "error": {"code": -1, "message": "Invalid command"},
            }
        else:
            result: CommandResultMessage = {
                **command_msg,
                "result": cmd_result,
            }
        # return the response to the client
        return web.json_response(result)

    async def _handle_cometd_client(  # noqa: PLR0912, C901
        self, request: web.Request
    ) -> web.Response:
        """
        Handle CometD request on the json CLI.

        https://github.com/Logitech/slimserver/blob/public/8.4/Slim/Web/Cometd.pm
        """
        logger = self.logger.getChild("cometd")
        # ruff: noqa: PLR0915
        clientid: str = ""
        response = []
        streaming = False
        long_poll = False
        json_msg: list[dict[str, Any]] = await request.json()
        # cometd message is an array of commands/messages
        for cometd_msg in json_msg:
            channel = cometd_msg.get("channel")
            # try to figure out clientid
            if not clientid:
                clientid = cometd_msg.get("clientId")
            if not clientid and channel == "/meta/handshake":
                # generate new clientid
                clientid = uuid1().hex
                self._cometd_clients[clientid] = CometDClient(
                    client_id=clientid,
                )
            elif not clientid and channel in ("/slim/subscribe", "/slim/request"):
                # pull clientId out of response channel
                clientid = cometd_msg["data"]["response"].split("/")[1]
            elif not clientid and channel == "/slim/unsubscribe":
                # pull clientId out of unsubscribe
                clientid = cometd_msg["data"]["unsubscribe"].split("/")[1]
            assert clientid, "No clientID provided"
            logger.debug(
                "Incoming message for channel '%s' - clientid: %s",
                channel,
                clientid,
            )

            # messageid is optional but if provided we must pass it along
            msgid = cometd_msg.get("id", "")

            if clientid not in self._cometd_clients:
                # If a client sends any request and we do not have a valid clid record
                # because the streaming connection has been lost for example, re-handshake them
                return web.json_response(
                    [
                        {
                            "id": msgid,
                            "channel": channel,
                            "clientId": None,
                            "successful": False,
                            "timestamp": time.strftime(
                                "%a, %d %b %Y %H:%M:%S %Z",
                                time.gmtime(),
                            ),
                            "error": "invalid clientId",
                            "advice": {
                                "reconnect": "handshake",
                                "interval": 2000,
                            },
                        },
                    ],
                )

            # get the cometd_client object for the clientid
            cometd_client = self._cometd_clients[clientid]
            cometd_client.last_seen = int(time.time())

            if channel == "/meta/handshake":
                # handshake message
                response.append(
                    {
                        "id": msgid,
                        "channel": channel,
                        "version": "1.0",
                        "supportedConnectionTypes": ["long-polling", "streaming"],
                        "clientId": clientid,
                        "successful": True,
                        "advice": {
                            # one of "none", "retry", "handshake"
                            "reconnect": "retry",
                            # initial interval is 0 to support long-polling's connect request
                            "interval": 0,
                            "timeout": 60000,
                        },
                    },
                )
                # playerid (mac) and uuid belonging to the client is sent in the ext field
                if player_id := cometd_msg.get("ext", {}).get("mac"):
                    cometd_client.player_id = player_id
                    if (uuid := cometd_msg.get("ext", {}).get("uuid")) and (
                        player := self.server.get_player(player_id)
                    ):
                        player.extra_data["uuid"] = uuid

            elif channel in ("/meta/connect", "/meta/reconnect"):
                # (re)connect message
                logger.debug("Client (re-)connected: %s", clientid)
                streaming = cometd_msg["connectionType"] == "streaming"
                long_poll = not streaming
                cometd_client.streaming = streaming
                # confirm the connection
                response.append(
                    {
                        "id": msgid,
                        "channel": channel,
                        "clientId": clientid,
                        "successful": True,
                        "timestamp": time.strftime(
                            "%a, %d %b %Y %H:%M:%S %Z",
                            time.gmtime(),
                        ),
                        "advice": {
                            # update interval for streaming mode
                            "interval": 5000 if streaming else 0,
                        },
                    },
                )
                # TODO: do we want to implement long-polling support too ?
                # https://github.com/Logitech/slimserver/blob/d9ebda7ebac41e82f1809dd85b0e4446e0c9be36/Slim/Web/Cometd.pm#L292

            elif channel == "/meta/disconnect":
                # disconnect message
                logger.debug("CometD Client disconnected: %s", clientid)
                self._cometd_clients.pop(clientid)
                return web.json_response(
                    [
                        {
                            "id": msgid,
                            "channel": channel,
                            "clientId": clientid,
                            "successful": True,
                            "timestamp": time.strftime(
                                "%a, %d %b %Y %H:%M:%S %Z",
                                time.gmtime(),
                            ),
                        },
                    ],
                )

            elif channel == "/meta/subscribe":
                cometd_client.meta_subscriptions.add(cometd_msg["subscription"])
                response.append(
                    {
                        "id": msgid,
                        "channel": channel,
                        "clientId": clientid,
                        "successful": True,
                        "subscription": cometd_msg["subscription"],
                    },
                )

            elif channel == "/meta/unsubscribe":
                if cometd_msg["subscription"] in cometd_client.meta_subscriptions:
                    cometd_client.meta_subscriptions.remove(cometd_msg["subscription"])
                response.append(
                    {
                        "id": msgid,
                        "channel": channel,
                        "clientId": clientid,
                        "successful": True,
                        "subscription": cometd_msg["subscription"],
                    },
                )
            elif channel == "/slim/subscribe":
                # ruff: noqa: E501, ERA001
                # A request to execute & subscribe to some Logitech Media Server event
                # A valid /slim/subscribe message looks like this:
                # {
                #   channel  => '/slim/subscribe',
                #   id       => <unique id>,
                #   data     => {
                #     response => '/slim/serverstatus', # the channel all messages should be sent back on
                #     request  => [ '', [ 'serverstatus', 0, 50, 'subscribe:60' ],
                #     priority => <value>, # optional priority value, is passed-through with the response
                #   }
                response.append(
                    {
                        "id": msgid,
                        "channel": channel,
                        "clientId": clientid,
                        "successful": True,
                    },
                )
                cometd_client.slim_subscriptions[cometd_msg["data"]["response"]] = (
                    cometd_msg
                )
                # Return one-off result now, rest is handled by the subscription logic
                self._handle_cometd_client_request(cometd_client, cometd_msg)

            elif channel == "/slim/unsubscribe":
                # ruff: noqa: E501, ERA001
                # A request to unsubscribe from a Logitech Media Server event, this is not the same as /meta/unsubscribe
                # A valid /slim/unsubscribe message looks like this:
                # {
                #   channel  => '/slim/unsubscribe',
                #   data     => {
                #     unsubscribe => '/slim/serverstatus',
                #   }
                response.append(
                    {
                        "id": msgid,
                        "channel": channel,
                        "clientId": clientid,
                        "successful": True,
                    },
                )
                cometd_client.slim_subscriptions.pop(
                    cometd_msg["data"]["unsubscribe"],
                    None,
                )

            elif channel == "/slim/request":
                # ruff: noqa: E501, ERA001
                # A request to execute a one-time Logitech Media Server event
                # A valid /slim/request message looks like this:
                # {
                #   channel  => '/slim/request',
                #   id       => <unique id>, (optional)
                #   data     => {
                #     response => '/slim/<clientId>/request',
                #     request  => [ '', [ 'menu', 0, 100, ],
                #     priority => <value>, # optional priority value, is passed-through with the response
                #   }
                if not msgid:
                    # If the caller does not want the response, id will be undef
                    logger.debug(
                        "Not sending response to request, caller does not want it",
                    )
                else:
                    # This response is optional, but we do it anyway
                    response.append(
                        {
                            "id": msgid,
                            "channel": channel,
                            "clientId": clientid,
                            "successful": True,
                        },
                    )
                    self._handle_cometd_client_request(cometd_client, cometd_msg)
            else:
                logger.warning("Unhandled channel %s", channel)
                # always reply with the (default) response to every message
                response.append(
                    {
                        "channel": channel,
                        "id": msgid,
                        "clientId": clientid,
                        "successful": True,
                    },
                )
        # append any remaining messages from the queue
        while True:
            try:
                msg = cometd_client.queue.get_nowait()
                response.append(msg)
            except asyncio.QueueEmpty:
                break
        # send response
        headers = {
            "Server": "Logitech Media Server (7.9.9 - 1667251155)",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Expires": "-1",
            "Connection": "keep-alive",
        }
        if not streaming:
            # Long-polling connect: if we don't already have queued data messages,
            # hold the connection open until a message arrives or timeout (30s).
            if long_poll and not any(
                msg for msg in response if msg.get("channel", "").startswith("/slim/")
            ):
                try:
                    msg = await asyncio.wait_for(cometd_client.queue.get(), timeout=30)
                    response.append(msg)
                    # drain any additional messages that arrived
                    while True:
                        try:
                            response.append(cometd_client.queue.get_nowait())
                        except asyncio.QueueEmpty:
                            break
                except (TimeoutError, asyncio.CancelledError):
                    pass
            cometd_client.last_seen = int(time.time())
            return web.json_response(response, headers=headers)

        # streaming mode: send messages from the queue to the client
        # the subscription connection is kept open and events are streamed to the client
        headers.update({"Content-Type": "application/json"})
        resp = web.StreamResponse(
            status=200,
            reason="OK",
            headers=headers,
        )
        resp.enable_chunked_encoding()
        await resp.prepare(request)
        chunk = json.dumps(response).encode("utf8")
        await resp.write(chunk)

        # keep delivering messages to the client until it disconnects
        # keep sending messages/events from the client's queue
        try:
            while True:
                # make sure we always send an array of messages
                msg = [await cometd_client.queue.get()]
                try:
                    chunk = json.dumps(msg).encode("utf8")
                    await resp.write(chunk)
                    cometd_client.last_seen = int(time.time())
                except (
                    ConnectionResetError,
                    ConnectionError,
                    BrokenPipeError,
                    RuntimeError,
                ):
                    break
        except asyncio.CancelledError:
            pass
        finally:
            # Clean up the client when the streaming connection ends
            logger.debug("CometD streaming connection ended for client: %s", clientid)
            if clientid in self._cometd_clients:
                client = self._cometd_clients.pop(clientid)
                empty_queue(client.queue)
        return resp

    def _handle_cometd_client_request(
        self,
        client: CometDClient,
        cometd_request: dict[str, Any],
    ) -> None:
        """
        Handle CometD request on the json CLI.

        https://github.com/Logitech/slimserver/blob/public/8.4/Slim/Web/Cometd.pm
        """

        async def _handle() -> None:
            try:
                result = await self._handle_command(cometd_request["data"]["request"])
                with suppress(asyncio.QueueFull):
                    client.queue.put_nowait(
                        {
                            "channel": cometd_request["data"]["response"],
                            "id": cometd_request["id"],
                            "data": result,
                            "ext": {"priority": cometd_request["data"].get("priority")},
                        },
                    )
            except asyncio.CancelledError:
                pass  # Task was cancelled, clean exit
            except Exception as err:  # noqa: BLE001
                self.logger.debug(
                    "Error handling CometD request: %s",
                    err,
                    exc_info=err,
                )

        asyncio.create_task(_handle())

    async def _handle_command(self, params: tuple[str, list[SlimCLIScalar]]) -> Any:
        """Handle a JSON-RPC or CometD command."""
        # Slim request handler
        # {"method":"slim.request","id":1,"params":["aa:aa:ca:5a:94:4c",["status","-", 2, "tags:xcfldatgrKN"]]}
        self.logger.debug("Handling request: %s", str(params))
        player_id = params[0]
        command = str(params[1][0])
        args, kwargs = parse_args(params[1][1:])
        slim_command = SlimCLICommand(
            player_id=player_id or None,
            command=command,
            args=args,
            kwargs=kwargs,
        )
        if (
            player_id
            and "seq_no" in kwargs
            and (player := self.server.get_player(player_id))
        ):
            player.extra_data["seq_no"] = int(kwargs["seq_no"])
        try:
            cmd_result = await self._dispatch_command(slim_command)
        except NotImplementedError:
            self._publish_unhandled_command(slim_command)
            return None
        if cmd_result is None:
            return {}
        if isinstance(cmd_result, (dict, list)):
            return cmd_result
        # Preserve the behaviour of legacy built-in handlers during the transition.
        # individual values are returned with underscore ?!
        return {f"_{command}": cmd_result}

    def _handle_players(
        self,
        player_id: str,
        start_index: int | str = 0,
        limit: int = 999,
        *args,  # Accept but discard any additional args
        **kwargs,
    ) -> PlayersResponse:
        """Handle players command."""
        players: list[PlayerItem] = []
        for index, player in enumerate(self.server.players):
            if not isinstance(start_index, int):
                start_index = 0
            if isinstance(start_index, int) and index < start_index:
                continue
            if len(players) >= limit:
                break
            item = create_player_item(index, player)
            item["name"] = self._display_name(player)
            players.append(item)
        return PlayersResponse(count=len(players), players_loop=players)

    def _display_name(self, player: SlimClient) -> str:
        """Real player name to show on the device's own screen and in.

        serverstatus/players_loop - prefers Music Assistant's own
        display_name (player.name if the user set a custom one via MA's
        own UI, else MA's own stored default_name, else the bare
        player_id - MA's own real display_name property, confirmed via
        its own source: "always prefer the custom name over the default
        name") over player.name here (aioslimproto's own SlimClient
        property, confirmed via its own source to be read-only and
        sourced purely from the device's own self-reported name at
        handshake, with no override mechanism of any kind - this
        project implements no "name" CLI command either, the real
        mechanism real LMS's own "Squeezebox Name" settings item uses
        to push a persistent override down to a player). A real device
        test confirmed the actual, user-visible consequence: renaming a
        player via MA's own UI had no effect on what the device itself
        displayed, even across a reboot - because nothing in this
        project ever read that renamed value in the first place.
        Falls back to aioslimproto's own player.name when this player
        isn't resolvable as a real MA player object (e.g. a sync-group
        member not individually registered) - same value as before,
        never worse than today's behavior.
        """
        if self.display_name_lookup is not None:
            name = self.display_name_lookup(player.player_id)
            if name:
                return name
        return player.name

    @staticmethod
    def _corrected_elapsed_seconds(player: SlimClient) -> float:
        """Real, stream-relative player.elapsed_seconds corrected for a.

        seeked stream's own position-0 restart. Confirmed via a real
        device test and PlayerMedia's own field comment (music_assistant_
        models/player.py: "length of the audio stream as delivered to
        the player, differs from duration after a seek on transports
        whose stream restarts at position zero" - exactly this
        project's own situation): after a seek, aioslimproto's own
        elapsed_seconds reports position within the NEW, shorter stream
        (0-based from the seek point), not position within the real
        track - the audio itself was already correct, but every
        elapsed-time value reported to the client needs this same
        correction, or the displayed progress bar/time resets to 0:00
        and counts up from there instead of showing the real position.
        player.current_media.metadata's own "elapsed_offset" (set in
        player.py's play_media(), alongside this same real field) is
        exactly that correction - how many seconds into the track the
        current stream's own position 0 actually corresponds to (0 when
        no seek has happened yet, since duration == stream_duration
        then).

        Guarded to PlayerState.STOPPED (real, confirmed value -
        models.py's own PLAYMODE_MAP) - returns a flat 0, confirmed
        correct by direct testing against real LMS hardware (a real
        device test found it resets to 0 once a track finishes, not
        the clamped end-of-track value StreamingController.pm's own
        playingSongElapsed() seemed to suggest - that source reading
        either doesn't apply to this exact scenario or missed something
        a different code path handles; direct hardware testing wins
        here). Two earlier, wrong guesses in this same spot: first
        returning player.elapsed_seconds directly for the stopped case
        (found frozen at a stale, seek-relative position, e.g. 11s),
        then briefly showing the clamped duration instead (real per
        source, but disproven by direct testing against real LMS
        itself) before settling on this flat 0.
        """
        if player.state == PlayerState.STOPPED:
            return 0
        offset = 0
        if player.current_media and player.current_media.metadata:
            offset = player.current_media.metadata.get("elapsed_offset", 0) or 0
        return player.elapsed_seconds + offset

    async def _handle_status(
        self,
        player_id: str,
        offset: int | str = "-",
        limit: int = 2,
        menu: str = "",
        useContextMenu: int | bool = False,  # noqa: N803
        tags: str = "xcfldatgrKN",
        **kwargs,
    ) -> PlayerStatusResponse:
        """Handle player status command."""
        player = self.server.get_player(player_id)
        if player is None:
            return None
        playlist_items: list[MediaDetails] = []
        if player.current_media:
            playlist_items.append(player.current_media)
        if player.next_media:
            playlist_items.append(player.next_media)
        # base details
        result = {
            "player_name": self._display_name(player),
            "player_connected": int(player.connected),
            "player_needs_upgrade": 0,
            "player_is_upgrading": 0,
            "power": int(player.powered),
            "signalstrength": 0,
            # "waitingToPlay" deliberately omitted here, not hardcoded to
            # 0 - confirmed via real LMS source (Slim/Control/Queries.pm's
            # own status handler): it's only ever conditionally added,
            # as 1, when the player isPlaying() but not yet "really"
            # playing (buffering before actual audio output starts) -
            # and left out of the response entirely otherwise, never
            # sent as 0. Root-caused as the actual reason the Now
            # Playing screen's elapsed/remaining time never displayed at
            # all (not just wrong values): real client source (jive/
            # slim/Player.lua's own _process_status) does
            # `self.waitingToPlay = event.data.waitingToPlay or false` -
            # Lua treats 0 as truthy (only nil/false are falsy), so a
            # sent "waitingToPlay":0 became `0 or false` = 0, which is
            # itself truthy - isWaitingToPlay() always returned true,
            # and _updatePosition()'s own real "Bug 15814: do not update
            # position if track isn't actually playing" guard
            # unconditionally returned early on every single call. This
            # project has no distinct "isPlaying but not really playing
            # yet" buffering state to report as 1, so omitting the key
            # entirely (matching real LMS's own "not waiting" case) is
            # the correct behavior here, not a partial fix.
        }
        cur_item = playlist_items[0] if playlist_items else None
        # Real LMS's own status handler (Slim/Control/Queries.pm's
        # statusQuery) NEVER gates any of this behind power state -
        # "power" is just another, independent field reported
        # alongside everything else here, not a gate for the rest of
        # the response. Confirmed directly from real source: mode is
        # added unconditionally; remote/current_title/time/rate/
        # duration/can_seek are gated on `if (my $song =
        # $client->playingSong())` - whether a song is actually
        # loaded, not on power; playlist_tracks is added
        # unconditionally, even when the queue is empty;
        # playlist_cur_index is gated only on `if ($songCount > 0)`.
        # This whole block used to be wrapped in "if player.powered:"
        # instead, which was the real, confirmed root cause of a
        # separate-looking bug: a real device test found the entire
        # queue (14 real items, correctly present in MA's own queue
        # the whole time) invisible on the client's Current Playlist
        # screen after a server boot, showing an empty-playlist
        # placeholder instead - because this player had never been
        # powered on, that gate silently withheld mode/
        # playlist_tracks/item_loop from every single status
        # response as a direct result, exactly matching what was
        # observed in a real device capture of this failure mode.
        result = {
            **result,
            "mode": PLAYMODE_MAP[player.state],
            **(
                {
                    "remote": 1,
                    "current_title": self.server.name,
                    "time": int(self._corrected_elapsed_seconds(player)),
                    # Confirmed via real LMS source (Slim/Control/Queries.pm's
                    # own status handler): hardcoded to 1 unconditionally,
                    # with a comment noting it's "just here for backward
                    # compatibility with older SBC firmware" - no trick-mode
                    # semantics to replicate, this project doesn't support
                    # trick modes either. Missing entirely until now - real
                    # client source (jive/slim/Player.lua's own
                    # _process_status) confirmed why that broke the Now
                    # Playing screen's elapsed/remaining time display
                    # completely rather than just showing a wrong value:
                    # self.rate = tonumber(event.data.rate) went to nil, and
                    # getTrackElapsed()'s own trick-mode correction
                    # (self.trackCorrection = self.rate * (now -
                    # self.trackSeen), only run while self.mode == "play")
                    # multiplies by it unconditionally - nil * a number
                    # errors out in Lua, taking the whole calculation down
                    # rather than degrading gracefully.
                    "rate": 1,
                    "duration": (_duration := cur_item.metadata.get("duration", 0)),
                    # Confirmed via real LMS source (same status handler,
                    # right after "duration" there too): conditionally added
                    # as 1 only when the current item can actually be
                    # seeked, never sent as 0 - same pattern as
                    # "waitingToPlay" above. Gated on _duration here since
                    # MA's own player_queues.seek() itself refuses to seek
                    # an item with no known duration ("Can not seek items
                    # without duration") - matches real LMS's own dependency
                    # (Slim::Music::Info::canSeek) closely enough without
                    # inventing a separate, unconfirmed seekability check of
                    # our own. Real client source (jive/slim/Player.lua's
                    # own isTrackSeekable()) reads this directly to decide
                    # whether the Now Playing screen's progress slider is
                    # enabled or shown disabled.
                    **({"can_seek": 1} if _duration else {}),
                }
                if cur_item
                else {}
            ),
            "sync_master": "",
            "sync_slaves": "",
            "mixer volume": player.volume_level,
            "player_ip": player.device_address,
            **({"playlist_cur_index": 0} if playlist_items else {}),
            "playlist_tracks": len(playlist_items),
            "playlist_loop": [
                _drop_empty_quality_fields(
                    playlist_item_from_media_details(index, item)
                )
                for index, item in enumerate(playlist_items)
            ],
            **player.extra_data,
        }

        # additional details if menu requested
        if menu == "menu":
            # in menu-mode the regular playlist_loop is replaced by item_loop
            result.pop("playlist_loop", None)
            preset_data: list[dict] = []
            preset_loop: list[int] = []
            for index, preset in enumerate(player.presets):
                preset_data.append(
                    {
                        "URL": str(index),
                        "text": preset.text,
                        "type": "audio",
                    },
                )
                preset_loop.append(1)

            while len(preset_loop) < 10:
                preset_data.append({})
                preset_loop.append(0)
            result = {
                **result,
                "alarm_state": "none",
                "alarm_snooze_seconds": 540,
                "alarm_timeout_seconds": 3600,
                "count": len(playlist_items),
                "offset": offset,
                "base": {
                    "actions": {
                        "more": {
                            "itemsParams": "params",
                            "window": {"isContextMenu": 1},
                            "cmd": ["contextmenu"],
                            "player": 0,
                            "params": {"context": "playlist", "menu": "track"},
                        },
                    },
                },
                "preset_loop": preset_loop,
                "preset_data": preset_data,
                "item_loop": [
                    menu_item_from_media_details(
                        item,
                    )
                    for item in (player.current_media, player.next_media)
                    if item
                ],
            }
        # additional details if contextmenu requested
        if bool(useContextMenu):
            result = {
                **result,
                # TODO ?!,
            }
        return result

    async def _handle_serverstatus(
        self,
        player_id: str,
        start_index: int = 0,
        limit: int = 2,
        *args,  # Accept but discard any additional args
        **kwargs,
    ) -> ServerStatusResponse:
        """Handle server status command."""
        # Devices sometimes send ['serverstatus', '-', '-', []]]
        if start_index == "-":
            start_index = 0
        if limit == "-":
            limit = float("inf")
        players: list[PlayerItem] = []
        for index, player in enumerate(self.server.players):
            if isinstance(start_index, int) and index < start_index:
                continue
            if len(players) > limit:
                break
            item = create_player_item(start_index + index, player)
            item["name"] = self._display_name(player)
            players.append(item)
        return ServerStatusResponse(
            {
                "httpport": self.cli_port_json,
                "ip": self.server.ip_address,
                "version": "7.999.999",
                "uuid": "aioslimproto",
                # TODO: set these vars ?
                "info total duration": 0,
                "info total genres": 0,
                "sn player count": 0,
                "lastscan": 1685548099,
                "info total albums": 0,
                "info total songs": 0,
                "info total artists": 0,
                "players_loop": players,
                "player count": len(players),
                "other player count": 0,
                "other_players_loop": [],
            },
        )

    async def _handle_firmwareupgrade(
        self,
        player_id: str,
        *args,
        **kwargs,
    ) -> ServerStatusResponse:
        """Handle firmwareupgrade command."""
        return {
            "firmwareUpgrade": 0,
            "relativeFirmwareUrl": "/firmware/baby_7.7.3_r16676.bin",
        }

    async def _handle_artworkspec(
        self,
        player_id: str,
        *args,
        **kwargs,
    ) -> ServerStatusResponse:
        """Handle firmwareupgrade command."""
        # https://github.com/Logitech/slimserver/blob/e9c2f88e7ca60b3648b66116240f3f5fe6ca3188/Slim/Control/Commands.pm#L224
        return None

    async def _handle_mixer(
        self,
        player_id: str,
        subcommand: str,
        *args,
        **kwargs,
    ) -> int | None:
        """Handle player mixer command."""
        arg = args[0] if args else None
        player = self.server.get_player(player_id)
        if not player:
            return None
        # <playerid> mixer volume <0 .. 100|-100 .. +100|?>
        if subcommand == "volume" and isinstance(arg, int) and arg >= 0:
            await player.volume_set(arg)
            return None
        if subcommand == "volume" and arg == "?":
            return player.volume_level
        if subcommand == "volume" and isinstance(arg, str) and "+" in arg:
            volume_level = min(100, player.volume_level + int(arg.split("+")[1]))
            await player.volume_set(volume_level)
            return None
        if subcommand == "volume" and isinstance(arg, int) and arg < 0:
            volume_level = max(0, player.volume_level + arg)
            await player.volume_set(volume_level)
            return None

        # <playerid> mixer muting <0|1|[toggle]|?|>
        if subcommand == "muting" and isinstance(arg, int):
            await player.mute(bool(arg))
            return None
        if subcommand == "muting" and (arg == "toggle" or arg is None):
            await player.mute(not player.muted)
            return None
        if subcommand == "muting" and arg == "?":
            return int(player.muted)
        raise NotImplementedError(f"No handler for mixer/{subcommand}")

    async def _handle_time(self, player_id: str, number: str | int) -> int | None:
        """Handle player `time` command."""
        # <playerid> time <number|-number|+number|?>
        # The "time" command allows you to query the current number of seconds that the
        # current song has been playing by passing in a "?".
        # You may jump to a particular position in a song by specifying a number of seconds
        # to seek to. You may also jump to a relative position within a song by putting an
        # explicit "-" or "+" character before a number of seconds you would like to seek.
        player = self.server.get_player(player_id)
        if not player:
            return None
        if number == "?":
            return int(self._corrected_elapsed_seconds(player))
        # Real, confirmed API (controllers/player_queues/controller.py's
        # own seek(queue_id, position) - queue_id is literally the
        # player_id, same pattern already confirmed/used elsewhere in
        # this project; position is an absolute number of seconds within
        # the current item, matching this command's own "jump to a
        # particular position" semantics exactly. Previously unimplemented
        # entirely (raised NotImplementedError for any real seek target,
        # only ever answering the "?" query above) - real device test
        # confirmed this as the actual cause of the seek bar "snapping
        # back" instead of seeking: the real client (jive/slim/Player.lua's
        # own gototime()) sends exactly this command when the Now
        # Playing screen's progress slider is dragged, and with no
        # handler the position never actually changed, so the next
        # status update simply reported the real, unchanged position.
        #
        # Relative seeks ("+N"/"-N", a string with an explicit sign
        # prefix per this command's own docstring above) are resolved
        # against the player's own current, corrected elapsed position
        # (not the raw, stream-relative one - a relative seek after an
        # earlier seek needs to add/subtract from where the track
        # actually is, not from the current stream's own position 0)
        # before calling seek() - MA's own seek() only takes an absolute
        # position, it has no relative-seek concept of its own.
        text = str(number)
        if text and text[0] in "+-":
            target = int(self._corrected_elapsed_seconds(player)) + int(text)
        else:
            target = int(number)
        # self.command_handler.mass, not self.mass or player.mass - two
        # real, confirmed-wrong guesses along the way here, both via
        # real docker log captures: self.mass doesn't exist on this
        # class (SlimProtoCLI only stores self.server, an aioslimproto
        # SlimServer) - AttributeError("'SlimProtoCLI' object has no
        # attribute 'mass'"); player.mass doesn't exist either, since
        # self.server.get_player(player_id) returns aioslimproto's own
        # native SlimClient (not this project's SqueezelitePlayer
        # wrapper, as assumed from player.elapsed_seconds alone working
        # - that's a real SlimClient property too, so it didn't actually
        # distinguish the two) - AttributeError("'SlimClient' object
        # has no attribute 'mass'"). The real, confirmed path came from
        # provider.py's own construction: SlimServer(cli_command_handler=
        # BrowseLibraryHandler(self), ...) - self here is the
        # SqueezelitePlayerProvider (real self.mass, inherited from
        # PlayerProvider). SlimProtoCLI.__init__ stores that handler as
        # self.command_handler, and BrowseLibraryHandler (browselibrary.py,
        # this same project) exposes .mass directly - the same object
        # every other handler in that file already uses throughout.
        await self.mass.player_queues.seek(queue_id=player_id, position=target)
        return None

    async def _handle_power(
        self,
        player_id: str,
        value: str | int,
        *args,
        **kwargs,
    ) -> int | None:
        """Handle player `time` command."""
        # <playerid> power <0|1|?|>
        # The "power" command turns the player on or off.
        # Use 0 to turn off, 1 to turn on, ? to query and
        # no parameter to toggle the power state of the player.
        player = self.server.get_player(player_id)
        if not player:
            return None
        if value == "?":
            return int(player.powered)
        await player.power(bool(value))
        return None

    async def _handle_play(
        self,
        player_id: str,
        *args,
        **kwargs,
    ) -> None:
        """Handle player `play` command.

        mass.player_queues.play(queue_id), not player.play() directly -
        confirmed via real source (aioslimproto/client.py's own
        SlimClient.play(): "if self._state != PlayerState.PAUSED: return"
        - a real device test found this explained "the play button
        doesn't do anything" precisely: once a track naturally ends,
        the player is PlayerState.STOPPED, not PAUSED, so player.play()
        silently no-ops every time. MA's own player_queues.play()
        (controllers/player_queues/controller.py) already handles both
        cases correctly: forwards to the same low-level player.play()
        when actually paused, or calls its own resume() otherwise -
        exactly the real "restart the current queue position" behavior
        this needed. self.command_handler.mass - the same real path
        confirmed for _handle_time's own seek() call above (see that
        method's own comment for the full account of how it was found).
        """
        await self.mass.player_queues.play(queue_id=player_id)

    async def _handle_stop(
        self,
        player_id: str,
        *args,
        **kwargs,
    ) -> None:
        """Handle player `stop` command."""
        if player := self.server.get_player(player_id):
            await player.stop()

    async def _handle_pause(
        self,
        player_id: str,
        force: int = 0,
        *args,
        **kwargs,
    ) -> None:
        """Handle player `stop` command.

        Same real fix as _handle_play above, same reason - this toggle's
        own "not currently playing" branch called the same low-level
        player.play() (a no-op unless actually paused), so toggling
        play/pause after a track naturally ended (PlayerState.STOPPED,
        not PAUSED) silently did nothing here too.
        """
        if player := self.server.get_player(player_id):
            if player.state == PlayerState.PLAYING:
                await player.pause()
            else:
                await self.mass.player_queues.play(queue_id=player_id)

    async def _handle_mode(
        self,
        player_id: str,
        subcommand: str,
        *args,
        **kwargs,
    ) -> None:
        """Handle player 'mode' command."""
        if subcommand == "play":
            return await self._handle_play(player_id, *args, **kwargs)
        if subcommand == "pause":
            return await self._handle_pause(player_id, *args, **kwargs)
        if subcommand == "stop":
            return await self._handle_stop(player_id, *args, **kwargs)

        raise NotImplementedError(f"No handler for mode/{subcommand}")

    async def _handle_button(
        self,
        player_id: str,
        subcommand: str,
        *args,
        **kwargs,
    ) -> None:
        """Handle player 'button' command."""
        player = self.server.get_player(player_id)
        if not player:
            return
        if subcommand == "volup":
            await player.volume_up()
            return
        if subcommand == "voldown":
            await player.volume_down()
            return
        if subcommand == "power":
            await player.power(not player.powered)
            return
        if subcommand == "jump_fwd" and player.next_media:
            await player.next()
            return
        if subcommand.startswith("preset_") and subcommand.endswith(".single"):
            # only handle http-based presets, ignore/forward all other
            preset_id = subcommand.split("preset_")[1].split(".", maxsplit=1)[0]
            preset_index = int(preset_id) - 1
            if len(player.presets) >= preset_index + 1:
                preset = player.presets[preset_index]
                if preset.uri.startswith("http"):
                    await player.play_url(
                        preset.uri,
                        metadata=MediaMetadata(
                            title=preset.text,
                            image_url=preset.icon,
                        ),
                    )

        raise NotImplementedError(f"No handler for button/{subcommand}")

    async def _handle_playlist(
        self,
        player_id: str,
        subcommand: str,
        *args,
        **kwargs,
    ) -> int | None:
        """Handle player `playlist` command."""
        # <playerid> playlist index <index|+index|-index|?> <fadeInSecs>
        arg = args[0] if args else "?"
        player = self.server.get_player(player_id)
        if not player:
            return None
        # we only handle playlist index +1 - the rest is forwarded
        if subcommand == "index" and arg in (1, "1", "+1") and player.next_media:
            await player.next()
            return None
        raise NotImplementedError(f"No handler for playlist/{subcommand}")

    async def _handle_menu(
        self,
        player_id: str,
        offset: int = 0,
        limit: int = 10,
        **kwargs,
    ) -> dict[str, Any]:
        """Handle menu request from CLI."""
        menu_items = []
        if player := self.server.get_player(player_id):
            for index, preset in enumerate(player.presets):
                preset_id = f"preset_{index + 1}"
                menu_items.append(
                    {
                        "id": preset_id,
                        "icon": preset.icon,
                        "text": preset.text,
                        "homeMenuText": preset.text,
                        "weight": 35,
                        "node": "myMusic",
                        "style": "itemplay",
                        "nextWindow": "nowPlaying",
                        "actions": {
                            "go": {
                                "cmd": ["button", f"{preset_id}.single"],
                                "itemsParams": "commonParams",
                                "params": {},
                                "player": 0,
                                "nextWindow": "nowPlaying",
                            },
                            "add": {
                                "player": 0,
                                "itemsParams": "commonParams",
                                "params": {"uri": preset.uri, "cmd": "add"},
                                "cmd": ["playlistcontrol"],
                                "nextWindow": "refresh",
                            },
                            "more": {
                                "player": 0,
                                "itemsParams": "commonParams",
                                "params": {"uri": preset.uri, "cmd": "add"},
                                "cmd": ["playlistcontrol"],
                                "nextWindow": "refresh",
                            },
                            "play": {
                                "cmd": ["playlistcontrol"],
                                "itemsParams": "commonParams",
                                "params": {
                                    "uri": preset.uri,
                                    "cmd": "play",
                                },
                                "player": 0,
                                "nextWindow": "nowPlaying",
                            },
                            "play-hold": {
                                "cmd": ["playlistcontrol"],
                                "itemsParams": "commonParams",
                                "params": {"uri": preset.uri, "cmd": "load"},
                                "player": 0,
                                "nextWindow": "nowPlaying",
                            },
                            "add-hold": {
                                "itemsParams": "commonParams",
                                "params": {"uri": preset.uri, "cmd": "insert"},
                                "player": 0,
                                "cmd": ["playlistcontrol"],
                                "nextWindow": "refresh",
                            },
                        },
                    },
                )
        return {
            "item_loop": menu_items[offset:limit],
            "offset": offset,
            "count": len(menu_items[offset:limit]),
        }

    def _handle_menustatus(
        self,
        player_id: str,
        *args,
        **kwargs,
    ) -> dict[str, Any]:
        """Handle menustatus request from CLI."""
        return None

    def _handle_displaystatus(
        self,
        player_id: str,
        *args,
        **kwargs,
    ) -> dict[str, Any]:
        """Handle displaystatus request from CLI."""
        return None

    def _handle_date(
        self,
        player_id: str,
        *args,
        **kwargs,
    ) -> dict[str, Any]:
        """Handle date request from CLI."""
        return {"date_epoch": int(time.time()), "date": "0000-00-00T00:00:00+00:00"}

    async def _on_player_event(self, event: SlimEvent) -> None:
        """Forward player events."""
        if not event.player_id:
            return
        client = next(
            (
                x
                for x in self._cometd_clients.values()
                if x.player_id == event.player_id
            ),
            None,
        )
        if not client:
            return
        # regular player updated (or connected) event, signal playerstatus
        if event.type in (EventType.PLAYER_CONNECTED, EventType.PLAYER_UPDATED):
            if sub := client.slim_subscriptions.get(
                f"/{client.client_id}/slim/playerstatus/{event.player_id}",
            ):
                self._handle_cometd_client_request(client, sub)
            if sub := client.slim_subscriptions.get(
                f"/{client.client_id}/slim/serverstatus",
            ):
                self._handle_cometd_client_request(client, sub)
            return
        # player presets updated, signal menustatus event
        if event.type == EventType.PLAYER_PRESETS_UPDATED and (
            sub := client.slim_subscriptions.get(
                f"/{client.client_id}/slim/menustatus/{event.player_id}",
            )
        ):
            menu = await self._handle_menu(event.player_id)
            with suppress(asyncio.QueueFull):
                client.queue.put_nowait(
                    {
                        "channel": sub["data"]["response"],
                        "id": sub["id"],
                        "data": [
                            event.player_id,
                            menu["item_loop"],
                            "replace",
                            event.player_id,
                        ],
                        "ext": {"priority": sub["data"].get("priority")},
                    },
                )

    def push_show_briefly(
        self,
        player_id: str,
        text: list[str],
        style: str | None = None,
        icon_id: str | None = None,
        duration_ms: int = 3000,
        kind: str = "mixed",
    ) -> None:
        """Push a real LMS "showBriefly" text/artwork popup - either the.

        KNOWN GAP: doesn't fire for MA-driven queue changes, only device-initiated ones - left as-is, too complex to fix cleanly.

        "mixed" kind (e.g. the "Adding" / "to play next..." confirmation
        with a badge and cover art shown when adding a track or album to
        the queue) or the "song" kind (the "Now Playing" + track title
        popup shown on every track load) - to a player's existing
        displaystatus subscription, on demand - not a response to a
        request, a push triggered at the moment an action (like
        playlistcontrol add/insert/load) completes.

        Confirmed real, end-to-end via LMS's own source, not guessed:
        - Real client source (jive/slim/Player.lua's own
          _process_displaystatus, subscribed via "displaystatus
          subscribe:showbriefly" at connect time) reads the pushed
          message as event.data.display directly. Its own type=="mixed"
          (or "popupalbum") branch - main text, a second "subtext" line,
          a badge icon ("style"=="add" -> badge_add), and artwork
          fetched via "icon-id" - is exactly what real LMS sends for a
          queue add. Its own type=="song" branch instead forwards the
          text straight to a "playerTitleStatus" notification (briefly
          overwriting the Now Playing screen's own title text) and
          shows no popup window at all ("showMe = false") - this is a
          DIFFERENT, separate real popup from the icon-only one
          push_play_icon (below) sends; LMS sends both, independently,
          for a single "load".
        - Real server source for the "mixed"/add case: Slim/Control/
          Commands.pm's own playlistcontrolCommand - `$client->
          showBriefly({'jive' => {'type' => 'mixed', 'style' => 'add',
          'text' => [$string, $info[0]], 'icon-id' => ...}})`.
        - Real server source for the "song"/load case: Slim/Player/
          StreamingController.pm's own _showTrackwaitStatus/
          _playersMessage - `type => 'song', text => [$line1, $line2],
          duration => 30000` for a local (non-remote) track. Initially
          assumed gated on the player having been stopped beforehand
          (matching that function's own "playingState == STOPPED"
          check) - disproven by direct device testing (fired even
          selecting "Play Now" on a track already playing); that field
          turned out to be StreamingController's own internal state
          machine, not the player's outwardly-visible mode, so this
          fires on every load, unconditionally.
        - Real strings.txt confirms the exact English text LMS itself
          uses for the "mixed"/add case: "Adding" (JIVE_POPUP_ADDING)
          and "to play next..." (JIVE_POPUP_TO_PLAY_NEXT) - callers
          pass whichever as text[0], with the track/album title as
          text[1]. For "song", real LMS uses "Now Playing" (the
          JIVE_POPUP... wait, actually NOW_PLAYING token) as text[0].

        This project's own _handle_displaystatus (the subscription
        request's own handler) deliberately still returns None -
        confirmed via real client source that a plain, un-pushed
        displaystatus subscription response carries no "display" key at
        all (the client only ever reads one when actually present), so
        there is nothing useful to answer the bare subscription request
        itself with. The real payload only exists at the moment this
        method is called.
        """
        display: dict[str, Any] = {"type": kind, "text": text, "duration": duration_ms}
        if style:
            display["style"] = style
        if icon_id:
            display["icon-id"] = icon_id
        self._push_displaystatus(player_id, display)

    def push_play_icon(
        self,
        player_id: str,
        text: list[str],
        icon_id: str | None = None,
    ) -> None:
        """Push the real, separate, icon-only "play" popup (a brief.

        ~1-2s icon with no visible text, shown on top of whatever screen
        is active right before the Now Playing screen appears) -
        confirmed via real LMS source as a SEPARATE showBriefly call
        from push_show_briefly's own "song" kind, both of which real LMS
        fires independently for the exact same "load a track" action:

        - Slim/Control/Commands.pm's own playcontrolCommand, handling a
          plain "play" command transitioning from a non-playing mode:
          its own comment confirms the real mechanism directly - "'play'
          from CLI needs to work the same as IR play button, by going
          through playlist jump - this will include a showBriefly to
          give feedback" - executing ['playlist', 'jump', $index,
          $fadeIn].
        - Slim/Control/Commands.pm's own playlistJumpCommand (the real
          handler for that 'playlist jump' command, and so also the
          real path a plain playlistcontrol cmd:load ultimately takes)
          calls a local $showStatus helper after actually starting
          playback: `$client->showBriefly($parts, {duration => 2})`
          where $parts comes from Slim/Player/Player.pm's own
          currentSongLines(jiveIconStyle => undef) - which defaults
          jiveIconStyle to the player's own current playmode ("play",
          immediately after starting playback) and builds `$jive = {
          'type' => 'icon', 'text' => [$status, $track->title], 'style'
          => $jiveIconStyle, 'play-mode' => $playmode, 'is-remote' =>
          $track->isRemoteURL}`.
        - Real client source (jive/slim/Player.lua's own
          _process_displaystatus) confirms exactly why this has no
          visible text despite the real payload carrying a "text"
          field: its own type=="icon"-with-style branch ("special")
          only ever sets the popup's CSS icon style
          ("icon_popup_"..style) - it never reads or displays
          display['text'] at all for this branch, unlike the "mixed"
          and "song" kinds above. Gated client-side to the player's own
          recent-input tracking ("icon-based showBrieflies only appear
          for IR" - Framework:isMostRecentInput('ir') or ('key')) -
          confirmed via real device testing that this condition is met
          for a context-menu "Play Now" selection on this project's own
          touchscreen hardware (JiveLite's own menu-selection handling
          registers as 'key' input internally), so this isn't something
          the server needs to account for - send the real payload
          either way and let the client's own, unchanged logic decide.

        "text" is a REQUIRED parameter, not optional, despite never
        being visibly displayed for this popup - confirmed via a real
        device test and a real client-side crash this caused before
        this parameter existed: _process_displaystatus (Player.lua)
        calls _formatShowBrieflyText(display['text']) UNCONDITIONALLY,
        before any branching on "type" at all - that function does
        `for i, v in ipairs(msg) do`, which raises "bad argument #1 to
        'ipairs' (table expected, got nil)" when msg is nil, aborting
        the entire response-sink callback right there - before the
        client ever reaches the code that would actually show the icon
        popup window. This is exactly why nothing appeared at all
        despite the server-side push itself succeeding completely (both
        push_show_briefly's and this method's own pushes confirmed
        reaching the client and being written to the wire via this
        project's own [DIAG] logging) - the crash happens entirely
        client-side, after a perfectly good push arrives. Callers should
        pass the same track-title text push_show_briefly's own "song"
        push already has on hand, for consistency with real LMS's own
        [$status, $track->title] shape, even though its contents are
        never actually seen.
        """
        display: dict[str, Any] = {
            "type": "icon",
            "style": "play",
            "play-mode": "play",
            "is-remote": 0,
            "text": text,
        }
        if icon_id:
            display["icon-id"] = icon_id
        self._push_displaystatus(player_id, display)

    def _push_displaystatus(self, player_id: str, display: dict[str, Any]) -> None:
        """Real, generic on-demand push to a player's existing.

        displaystatus subscription - factored out of push_show_briefly
        (the only caller until push_play_icon above needed the exact
        same mechanism for a differently-shaped payload).

        Mirrors _on_player_event's own existing menustatus push above
        (same client lookup, same subscription-dict shape, same raw
        client.queue.put_nowait(...) call) - the one real, structural
        difference is the pushed "data" shape itself: menustatus pushes
        a positional array ([player_id, item_loop, "replace",
        player_id], that command's own real response convention, not
        reusable here), where displaystatus pushes a plain {"display":
        {...}} dict - confirmed via this same file's own
        _handle_cometd_client_request, which wraps a command handler's
        return value directly as "data" with no transformation, so a
        normal (non-pushed) displaystatus response would carry exactly
        this same {"display": {...}} shape too.
        """
        client = next(
            (x for x in self._cometd_clients.values() if x.player_id == player_id),
            None,
        )
        if not client:
            return
        sub = client.slim_subscriptions.get(
            f"/{client.client_id}/slim/displaystatus/{player_id}",
        )
        if not sub:
            return
        with suppress(asyncio.QueueFull):
            client.queue.put_nowait(
                {
                    "channel": sub["data"]["response"],
                    "id": sub["id"],
                    "data": {"display": display},
                    "ext": {"priority": sub["data"].get("priority")},
                },
            )

    async def _do_periodic(self) -> None:
        """Execute periodic sending of state and cleanup."""
        while True:
            # cleanup orphaned clients
            disconnected_clients = set()
            for cometd_client in self._cometd_clients.values():
                stale_threshold = 80 if cometd_client.streaming else 45
                if (time.time() - cometd_client.last_seen) > stale_threshold:
                    disconnected_clients.add(cometd_client.client_id)
                    continue
            for clientid in disconnected_clients:
                client = self._cometd_clients.pop(clientid)
                empty_queue(client.queue)
                self.logger.debug("Cleaned up disconnected CometD Client: %s", clientid)
            # handle client subscriptions (only for streaming clients;
            # long-polling clients get subscription data on their next connect)
            for cometd_client in self._cometd_clients.values():
                if not cometd_client.streaming:
                    continue
                for sub in cometd_client.slim_subscriptions.values():
                    self._handle_cometd_client_request(cometd_client, sub)

            await asyncio.sleep(60)


def dict_to_strings(source: dict) -> list[str]:
    """Convert dict to key:value strings (used in slimproto cli)."""
    result: list[str] = []

    for key, value in source.items():
        if value in (None, ""):
            continue
        if isinstance(value, list):
            for subval in value:
                if isinstance(subval, dict):
                    result += dict_to_strings(subval)
                else:
                    result.append(str(subval))
        elif isinstance(value, dict):
            result += dict_to_strings(value)
        else:
            result.append(f"{key}:{value!s}")
    return result


def menu_item_from_media_details(
    media_item: MediaDetails,
    include_actions: bool = False,
) -> PlaylistItem:
    """Parse (menu) MediaItem from MA MediaItem."""
    go_action = {
        "cmd": ["playlistcontrol"],
        "itemsParams": "commonParams",
        "params": {"uri": media_item.url, "cmd": "play"},
        "player": 0,
        "nextWindow": "nowPlaying",
    }
    details = SlimMenuItem(
        track=media_item.metadata.get("title", media_item.url),
        album=media_item.metadata.get("album", ""),
        trackType="radio",
        icon=media_item.metadata.get("image_url", ""),
        artist=media_item.metadata.get("artist", ""),
        text=media_item.metadata.get("title", media_item.url),
        params={
            "track_id": media_item.metadata.get("item_id", media_item.url),
            "item_id": media_item.metadata.get("item_id", media_item.url),
            "uri": media_item.url,
        },
        type="track",
    )
    # optionally include actions
    if include_actions:
        details["actions"] = {
            "go": go_action,
            "add": {
                "player": 0,
                "itemsParams": "commonParams",
                "params": {"uri": media_item.url, "cmd": "add"},
                "cmd": ["playlistcontrol"],
                "nextWindow": "refresh",
            },
            "more": {
                "player": 0,
                "itemsParams": "commonParams",
                "params": {"uri": media_item.url, "cmd": "add"},
                "cmd": ["playlistcontrol"],
                "nextWindow": "refresh",
            },
            "play": {
                "cmd": ["playlistcontrol"],
                "itemsParams": "commonParams",
                "params": {
                    "uri": media_item.url,
                    "cmd": "play",
                },
                "player": 0,
                "nextWindow": "nowPlaying",
            },
            "play-hold": {
                "cmd": ["playlistcontrol"],
                "itemsParams": "commonParams",
                "params": {"uri": media_item.url, "cmd": "load"},
                "player": 0,
                "nextWindow": "nowPlaying",
            },
            "add-hold": {
                "itemsParams": "commonParams",
                "params": {"uri": media_item.url, "cmd": "insert"},
                "player": 0,
                "cmd": ["playlistcontrol"],
                "nextWindow": "refresh",
            },
        }
    details["style"] = "itemplay"
    details["nextWindow"] = "nowPlaying"
    return details


# cli.py patch v5
# v5 fixes player_name/name (status, players, serverstatus) always
# showing the device's own raw self-reported name (aioslimproto's
# player.name - read-only, sourced purely from the device's handshake,
# no override mechanism) regardless of any name set via Music
# Assistant's own UI. Real device test confirmed the actual symptom: a
# rename via MA's UI had zero effect on the device's own screen, even
# across a reboot. Now reads MA's own player.display_name via a new
# _display_name() helper, falling back to player.name when the player
# isn't resolvable as a real MA player object - never worse than
# before. Also: this file still implements no "name" CLI command (the
# real mechanism real LMS's own "Squeezebox Name" settings item uses
# to push a persistent rename FROM the device) - out of scope for this
# fix, which only addresses MA-side renames propagating down.
# v1 fixed bitrate/samplerate/samplesize being hardcoded to "". v2 added
# "type". v3 stopped always including these four keys, to fix a real
# device-confirmed stray "* bits" (Lua treats "" as truthy) - but did
# that by changing playlist_item_from_media_details() itself, a public,
# unprefixed function whose own PlaylistItem return type declares these
# as required, so v3 risked a real KeyError for any other caller of
# this shared aioslimproto function using item["bitrate"] rather than
# .get(). v4 reworks this: playlist_item_from_media_details() is back
# to always returning all four (its original, fully backward-compatible
# contract), and the empty-key-dropping now happens in a new, separate
# _drop_empty_quality_fields() helper applied only at this file's own
# "status" response call site - the one place that's genuinely ours to
# shape, not the shared building block other projects might depend on.
# Local patch marker for this project - not an upstream aioslimproto
# version. Bump this comment (v5, v6, ...) on any further change to
# this file so a diff against a fresh pip download always shows what's
# actually been touched, same convention as browselibrary.py's own
# version marker.


def playlist_item_from_media_details(index: int, media: MediaDetails) -> PlaylistItem:
    """Parse PlaylistItem for the Json RPC interface from MediaDetails."""
    return {
        "playlist index": index,
        "id": "-187651250107376",
        "url": media.url,
        "title": media.metadata.get("title", media.url),
        "artist": media.metadata.get("artist", ""),
        "album": media.metadata.get("album", ""),
        "remote": 1,
        "artwork_url": media.metadata.get("image_url", ""),
        "coverid": "-187651250107376",
        "duration": media.metadata.get("duration", ""),
        # Previously hardcoded to "" unconditionally, regardless of what
        # MediaMetadata actually carried - these were never wired to
        # anything. That's the actual root cause of the blank/wrong
        # "44100 HZ * 32 bits" Now Playing display for MA-sourced tracks
        # (vs. real LMS's own "mp3 * 192kbps CBR * 44100hz"). Now sourced
        # the same way every other field in this function already is:
        # from the metadata dict the caller builds. Falls back to "" -
        # today's behavior - when a caller doesn't set these keys, so
        # this is additive-only and doesn't change behavior for any
        # existing MediaDetails producer.
        #
        # Deliberately kept as an unconditional "" fallback here, same
        # as every other field in this function, even though a real
        # device test found a client-side issue with that for these
        # four specifically (Lua treats "" as truthy, so a client-side
        # "if item.samplesize then ..." check still passes on an empty
        # string - see the real fix, applied at this function's one
        # call site instead of here). This function is public and
        # unprefixed - a real, shared piece of aioslimproto other
        # projects could import and call directly, and PlaylistItem
        # (models.py) declares all four of these as required str
        # fields, not NotRequired, so code elsewhere may reasonably use
        # item["bitrate"] rather than .get(). Changing this function's
        # own contract to sometimes omit keys would risk a real
        # KeyError for exactly that kind of caller. The Lua-specific
        # workaround belongs only where it's genuinely ours to shape:
        # the one place this function's result gets embedded into an
        # actual wire response, not in this shared building block
        # itself.
        #
        # "type" confirmed via real LMS source (Slim/Control/Queries.pm's
        # own CLI tag table: 'o' => ['type', 'TYPE', 'content_type']) as
        # a real, separate field from "bitrate" - the client concatenates
        # them itself (type * bitrate * samplerate), it doesn't derive
        # the codec name from anywhere else.
        "type": media.metadata.get("type", ""),
        "bitrate": media.metadata.get("bitrate", ""),
        "samplerate": media.metadata.get("samplerate", ""),
        "samplesize": media.metadata.get("samplesize", ""),
    }


def _drop_empty_quality_fields(item: PlaylistItem) -> PlaylistItem:
    """Drop empty type/bitrate/samplerate/samplesize keys before they hit the wire.

    Lua treats an empty string as truthy (only nil/false are falsy), so
    sending these with a "" value still passes a client-side check like
    "if item.samplesize then ...", leaving a stray "* bits" with no
    number - confirmed via a real device test. Applied only here, at
    the one call site that assembles the actual "status" response
    (below), not inside playlist_item_from_media_details() itself -
    that function is a public, unprefixed piece of shared aioslimproto
    other projects could call directly, and its own PlaylistItem return
    type (models.py) declares these four as required, so changing what
    it returns could break a caller using item["bitrate"] rather than
    .get(). This wrapper only reshapes what this local patch sends over
    the wire for its own response, without touching that shared
    function's contract for anyone else.
    """
    return {  # type: ignore[typeddict-item]
        k: v
        for k, v in item.items()
        if k not in ("type", "bitrate", "samplerate", "samplesize") or v
    }


def create_player_item(playerindex: int, player: SlimClient) -> PlayerItem:
    """Parse PlayerItem for the Json RPC interface from SlimClient."""
    return {
        "playerindex": str(playerindex),
        "playerid": player.player_id,
        "name": player.name,
        "modelname": player.device_model,
        "connected": int(player.connected),
        "isplaying": 1 if player.state == PlayerState.PLAYING else 0,
        "power": int(player.powered),
        "model": player.device_type,
        "canpoweroff": 1,
        "firmware": "unknown",
        "isplayer": 1,
        "displaytype": "none",
        "uuid": player.extra_data["uuid"],
        "seq_no": str(player.extra_data["seq_no"]),
        "ip": player.device_address,
    }
