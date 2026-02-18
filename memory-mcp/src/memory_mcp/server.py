"""MCP Server for AI Long-term Memory - Let AI remember across sessions!"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .config import MemoryConfig, ServerConfig
from .episode import EpisodeManager
from .memory import MemoryStore
from .sensory import SensoryIntegration
from .types import CameraPosition

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryMCPServer:
    """MCP Server that gives AI long-term memory."""

    def __init__(self):
        self._server = Server("memory-mcp")
        self._memory_store: MemoryStore | None = None
        self._episode_manager: EpisodeManager | None = None  # Phase 4.2
        self._sensory_integration: SensoryIntegration | None = None  # Phase 4.3
        self._server_config = ServerConfig.from_env()
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Set up MCP tool handlers."""

        @self._server.list_tools()
        async def list_tools() -> list[Tool]:
            """List available memory tools."""
            return [
                Tool(
                    name="remember",
                    description="Save a memory to long-term storage. Use this to remember important things, experiences, conversations, or learnings.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The memory content to save",
                            },
                            "emotion": {
                                "type": "string",
                                "description": "Emotion associated with this memory",
                                "default": "neutral",
                                "enum": ["happy", "sad", "surprised", "moved", "excited", "nostalgic", "curious", "neutral"],
                            },
                            "importance": {
                                "type": "integer",
                                "description": "Importance level from 1 (trivial) to 5 (critical)",
                                "default": 3,
                                "minimum": 1,
                                "maximum": 5,
                            },
                            "category": {
                                "type": "string",
                                "description": "Category of memory",
                                "default": "daily",
                                "enum": ["daily", "philosophical", "technical", "memory", "observation", "feeling", "conversation"],
                            },
                            "auto_link": {
                                "type": "boolean",
                                "description": "Automatically link to similar existing memories",
                                "default": True,
                            },
                            "link_threshold": {
                                "type": "number",
                                "description": "Similarity threshold for auto-linking (0-2, lower means more similar required)",
                                "default": 0.8,
                                "minimum": 0,
                                "maximum": 2,
                            },
                        },
                        "required": ["content"],
                    },
                ),
                Tool(
                    name="search_memories",
                    description="Search through memories using semantic similarity. Find memories related to a topic or query.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query to find related memories",
                            },
                            "n_results": {
                                "type": "integer",
                                "description": "Maximum number of results to return",
                                "default": 5,
                                "minimum": 1,
                                "maximum": 20,
                            },
                            "emotion_filter": {
                                "type": "string",
                                "description": "Filter by emotion (optional)",
                                "enum": ["happy", "sad", "surprised", "moved", "excited", "nostalgic", "curious", "neutral"],
                            },
                            "category_filter": {
                                "type": "string",
                                "description": "Filter by category (optional)",
                                "enum": ["daily", "philosophical", "technical", "memory", "observation", "feeling", "conversation"],
                            },
                            "date_from": {
                                "type": "string",
                                "description": "Filter memories from this date (ISO 8601 format, optional)",
                            },
                            "date_to": {
                                "type": "string",
                                "description": "Filter memories until this date (ISO 8601 format, optional)",
                            },
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="recall",
                    description="Automatically recall relevant memories based on the current conversation context. Use this to remember things that might be relevant.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "context": {
                                "type": "string",
                                "description": "Current conversation context or topic",
                            },
                            "n_results": {
                                "type": "integer",
                                "description": "Number of memories to recall",
                                "default": 3,
                                "minimum": 1,
                                "maximum": 10,
                            },
                        },
                        "required": ["context"],
                    },
                ),
                Tool(
                    name="list_recent_memories",
                    description="List the most recent memories. Use this to see what has been remembered recently.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of memories to list",
                                "default": 10,
                                "minimum": 1,
                                "maximum": 50,
                            },
                            "category_filter": {
                                "type": "string",
                                "description": "Filter by category (optional)",
                                "enum": ["daily", "philosophical", "technical", "memory", "observation", "feeling", "conversation"],
                            },
                        },
                        "required": [],
                    },
                ),
                Tool(
                    name="get_memory_stats",
                    description="Get statistics about stored memories. Shows total count, breakdown by category and emotion.",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                ),
                Tool(
                    name="recall_with_associations",
                    description="Recall memories with their associated/linked memories. Returns the primary memories plus any memories linked to them.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "context": {
                                "type": "string",
                                "description": "Current context or topic",
                            },
                            "n_results": {
                                "type": "integer",
                                "description": "Number of primary memories to recall",
                                "default": 3,
                                "minimum": 1,
                                "maximum": 10,
                            },
                            "chain_depth": {
                                "type": "integer",
                                "description": "How many levels of links to follow (1-3)",
                                "default": 1,
                                "minimum": 1,
                                "maximum": 3,
                            },
                        },
                        "required": ["context"],
                    },
                ),
                Tool(
                    name="recall_divergent",
                    description="Recall memories with divergent associative thinking. Expands memory candidates and selects them through workspace-style competition.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "context": {
                                "type": "string",
                                "description": "Current conversation context or topic",
                            },
                            "n_results": {
                                "type": "integer",
                                "description": "Number of memories to recall",
                                "default": 5,
                                "minimum": 1,
                                "maximum": 20,
                            },
                            "max_branches": {
                                "type": "integer",
                                "description": "Maximum branches per node during associative expansion",
                                "default": 3,
                                "minimum": 1,
                                "maximum": 8,
                            },
                            "max_depth": {
                                "type": "integer",
                                "description": "Maximum depth during associative expansion",
                                "default": 3,
                                "minimum": 1,
                                "maximum": 5,
                            },
                            "temperature": {
                                "type": "number",
                                "description": "Selection temperature (lower is more focused)",
                                "default": 0.7,
                                "minimum": 0.1,
                                "maximum": 2.0,
                            },
                            "include_diagnostics": {
                                "type": "boolean",
                                "description": "Include diagnostic metrics in the output",
                                "default": False,
                            },
                        },
                        "required": ["context"],
                    },
                ),
                Tool(
                    name="get_association_diagnostics",
                    description="Inspect associative expansion diagnostics for a given context without committing activation updates.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "context": {
                                "type": "string",
                                "description": "Context used to probe associative expansion",
                            },
                            "sample_size": {
                                "type": "integer",
                                "description": "Sample size for diagnostic probing",
                                "default": 20,
                                "minimum": 3,
                                "maximum": 20,
                            },
                        },
                        "required": ["context"],
                    },
                ),
                Tool(
                    name="consolidate_memories",
                    description="Run a manual replay/consolidation cycle to strengthen associations and refresh activation metadata.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "window_hours": {
                                "type": "integer",
                                "description": "Look-back window for replay candidates in hours",
                                "default": 24,
                                "minimum": 1,
                                "maximum": 168,
                            },
                            "max_replay_events": {
                                "type": "integer",
                                "description": "Maximum replay transitions to process",
                                "default": 200,
                                "minimum": 1,
                                "maximum": 1000,
                            },
                            "link_update_strength": {
                                "type": "number",
                                "description": "Strength for coactivation/link updates",
                                "default": 0.2,
                                "minimum": 0.01,
                                "maximum": 1.0,
                            },
                        },
                        "required": [],
                    },
                ),
                Tool(
                    name="get_memory_chain",
                    description="Get a memory and all memories linked to it. Useful for exploring related memories.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "memory_id": {
                                "type": "string",
                                "description": "ID of the starting memory",
                            },
                            "depth": {
                                "type": "integer",
                                "description": "How deep to follow links",
                                "default": 2,
                                "minimum": 1,
                                "maximum": 5,
                            },
                        },
                        "required": ["memory_id"],
                    },
                ),
                # Phase 4: Episode Memory Tools
                Tool(
                    name="create_episode",
                    description="Create an episode from recent memories. Use this to group related experiences into a story (e.g., 'Morning sky search').",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "Episode title (e.g., 'Morning sky search')",
                            },
                            "memory_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of memory IDs to include in the episode",
                            },
                            "participants": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "People involved in the episode (optional)",
                                "default": [],
                            },
                            "auto_summarize": {
                                "type": "boolean",
                                "description": "Auto-generate summary from memories",
                                "default": True,
                            },
                        },
                        "required": ["title", "memory_ids"],
                    },
                ),
                Tool(
                    name="search_episodes",
                    description="Search through past episodes. Find a sequence of experiences by topic.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for episodes",
                            },
                            "n_results": {
                                "type": "integer",
                                "description": "Maximum number of results",
                                "default": 5,
                                "minimum": 1,
                                "maximum": 20,
                            },
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="get_episode_memories",
                    description="Get all memories in a specific episode, in chronological order.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "episode_id": {
                                "type": "string",
                                "description": "Episode ID",
                            },
                        },
                        "required": ["episode_id"],
                    },
                ),
                # Phase 4.3: Sensory Integration Tools
                Tool(
                    name="save_visual_memory",
                    description="Save a memory with visual data (image path and camera position). Use this when you see something with your camera.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "Memory content (e.g., 'Found the morning sky')",
                            },
                            "image_path": {
                                "type": "string",
                                "description": "Path to the captured image file",
                            },
                            "camera_position": {
                                "type": "object",
                                "description": "Camera pan/tilt position",
                                "properties": {
                                    "pan_angle": {
                                        "type": "integer",
                                        "description": "Pan angle (-90 to +90)",
                                    },
                                    "tilt_angle": {
                                        "type": "integer",
                                        "description": "Tilt angle (-90 to +90)",
                                    },
                                    "preset_id": {
                                        "type": "string",
                                        "description": "Preset ID (optional)",
                                    },
                                },
                                "required": ["pan_angle", "tilt_angle"],
                            },
                            "emotion": {
                                "type": "string",
                                "description": "Emotion",
                                "default": "neutral",
                                "enum": ["happy", "sad", "surprised", "moved", "excited", "nostalgic", "curious", "neutral"],
                            },
                            "importance": {
                                "type": "integer",
                                "description": "Importance (1-5)",
                                "default": 3,
                                "minimum": 1,
                                "maximum": 5,
                            },
                            "resolution": {
                                "type": "string",
                                "description": "Image resolution for memory storage: 'low' (160x120), 'medium' (320x240, default), 'high' (640x480)",
                                "default": "medium",
                                "enum": ["low", "medium", "high"],
                            },
                        },
                        "required": ["content", "image_path", "camera_position"],
                    },
                ),
                Tool(
                    name="save_audio_memory",
                    description="Save a memory with audio data (audio file path and transcript). Use this when you hear something.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "Memory content (e.g., 'Heard a greeting')",
                            },
                            "audio_path": {
                                "type": "string",
                                "description": "Path to the audio file",
                            },
                            "transcript": {
                                "type": "string",
                                "description": "Transcribed text from audio (e.g., from Whisper)",
                            },
                            "emotion": {
                                "type": "string",
                                "description": "Emotion",
                                "default": "neutral",
                                "enum": ["happy", "sad", "surprised", "moved", "excited", "nostalgic", "curious", "neutral"],
                            },
                            "importance": {
                                "type": "integer",
                                "description": "Importance (1-5)",
                                "default": 3,
                                "minimum": 1,
                                "maximum": 5,
                            },
                        },
                        "required": ["content", "audio_path", "transcript"],
                    },
                ),
                Tool(
                    name="recall_by_camera_position",
                    description="Recall memories by camera direction. Find what you saw when looking in a specific direction.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "pan_angle": {
                                "type": "integer",
                                "description": "Pan angle (-90 to +90)",
                            },
                            "tilt_angle": {
                                "type": "integer",
                                "description": "Tilt angle (-90 to +90)",
                            },
                            "tolerance": {
                                "type": "integer",
                                "description": "Angle tolerance (default ±15 degrees)",
                                "default": 15,
                                "minimum": 1,
                                "maximum": 90,
                            },
                        },
                        "required": ["pan_angle", "tilt_angle"],
                    },
                ),
                # Phase 4.4: Working Memory Tools
                Tool(
                    name="get_working_memory",
                    description="Get recent memories from working memory buffer (fast access). Use this to quickly recall what just happened.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "n_results": {
                                "type": "integer",
                                "description": "Number of recent memories to get",
                                "default": 10,
                                "minimum": 1,
                                "maximum": 20,
                            },
                        },
                        "required": [],
                    },
                ),
                Tool(
                    name="refresh_working_memory",
                    description="Refresh working memory with important and frequently accessed memories from long-term storage.",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                ),
                # Phase 5: Causal Links
                Tool(
                    name="link_memories",
                    description="Create a causal or relational link between two memories. Use this to record 'A caused B' or 'A leads to B' relationships.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "source_id": {
                                "type": "string",
                                "description": "ID of the source memory",
                            },
                            "target_id": {
                                "type": "string",
                                "description": "ID of the target memory",
                            },
                            "link_type": {
                                "type": "string",
                                "description": "Type of link",
                                "default": "caused_by",
                                "enum": ["similar", "caused_by", "leads_to", "related"],
                            },
                            "note": {
                                "type": "string",
                                "description": "Optional note explaining the link",
                            },
                        },
                        "required": ["source_id", "target_id"],
                    },
                ),
                Tool(
                    name="get_causal_chain",
                    description="Trace the causal chain of a memory. Find what caused this memory (backward) or what it led to (forward).",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "memory_id": {
                                "type": "string",
                                "description": "ID of the starting memory",
                            },
                            "direction": {
                                "type": "string",
                                "description": "Direction to trace: 'backward' (find causes) or 'forward' (find effects)",
                                "default": "backward",
                                "enum": ["backward", "forward"],
                            },
                            "max_depth": {
                                "type": "integer",
                                "description": "How deep to trace the chain (1-5)",
                                "default": 3,
                                "minimum": 1,
                                "maximum": 5,
                            },
                        },
                        "required": ["memory_id"],
                    },
                ),
                Tool(
                    name="tom",
                    description="Theory of Mind: perspective-taking tool. Call this BEFORE responding to understand what the other person is feeling and wanting. Projects your simulated emotions onto them, then swaps perspectives.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "situation": {
                                "type": "string",
                                "description": "What the other person said or did (their message/action)",
                            },
                            "person": {
                                "type": "string",
                                "description": "Who you are talking to (default: コウタ)",
                                "default": "コウタ",
                            },
                        },
                        "required": ["situation"],
                    },
                ),
            ]

        @self._server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            """Handle tool calls."""
            if self._memory_store is None:
                return [TextContent(type="text", text="Error: Memory store not connected")]

            try:
                match name:
                    case "remember":
                        content = arguments.get("content", "")
                        if not content:
                            return [TextContent(type="text", text="Error: content is required")]

                        auto_link = arguments.get("auto_link", True)

                        if auto_link:
                            memory = await self._memory_store.save_with_auto_link(
                                content=content,
                                emotion=arguments.get("emotion", "neutral"),
                                importance=arguments.get("importance", 3),
                                category=arguments.get("category", "daily"),
                                link_threshold=arguments.get("link_threshold", 0.8),
                            )
                            linked_ids = await self._memory_store._store.get_linked_memory_ids(memory.id)
                            linked_info = f"\nLinked to: {len(linked_ids)} memories"
                        else:
                            memory = await self._memory_store.save(
                                content=content,
                                emotion=arguments.get("emotion", "neutral"),
                                importance=arguments.get("importance", 3),
                                category=arguments.get("category", "daily"),
                            )
                            linked_info = ""

                        return [
                            TextContent(
                                type="text",
                                text=f"Memory saved!\nID: {memory.id}\nTimestamp: {memory.timestamp}\nEmotion: {memory.emotion}\nImportance: {memory.importance}\nCategory: {memory.category}{linked_info}",
                            )
                        ]

                    case "search_memories":
                        query = arguments.get("query", "")
                        if not query:
                            return [TextContent(type="text", text="Error: query is required")]

                        results = await self._memory_store.search(
                            query=query,
                            n_results=arguments.get("n_results", 5),
                            emotion_filter=arguments.get("emotion_filter"),
                            category_filter=arguments.get("category_filter"),
                            date_from=arguments.get("date_from"),
                            date_to=arguments.get("date_to"),
                        )

                        if not results:
                            return [TextContent(type="text", text="No memories found matching the query.")]

                        output_lines = [f"Found {len(results)} memories:\n"]
                        for i, result in enumerate(results, 1):
                            m = result.memory
                            image_line = ""
                            for sd in m.sensory_data:
                                if sd.sensory_type == "visual" and sd.image_data:
                                    image_line = f"Image: data:image/jpeg;base64,{sd.image_data}\n"
                                    break
                            output_lines.append(
                                f"--- Memory {i} (distance: {result.distance:.4f}) ---\n"
                                f"ID: {m.id}\n"
                                f"[{m.timestamp}] [{m.emotion}] [{m.category}] (importance: {m.importance})\n"
                                f"{m.content}\n"
                                f"{image_line}"
                            )

                        return [TextContent(type="text", text="\n".join(output_lines))]

                    case "recall":
                        context = arguments.get("context", "")
                        if not context:
                            return [TextContent(type="text", text="Error: context is required")]

                        results = await self._memory_store.recall(
                            context=context,
                            n_results=arguments.get("n_results", 3),
                        )

                        if not results:
                            return [TextContent(type="text", text="No relevant memories found.")]

                        output_lines = [f"Recalled {len(results)} relevant memories:\n"]
                        for i, result in enumerate(results, 1):
                            m = result.memory
                            image_line = ""
                            for sd in m.sensory_data:
                                if sd.sensory_type == "visual" and sd.image_data:
                                    image_line = f"Image: data:image/jpeg;base64,{sd.image_data}\n"
                                    break
                            output_lines.append(
                                f"--- Memory {i} ---\n"
                                f"ID: {m.id}\n"
                                f"[{m.timestamp}] [{m.emotion}]\n"
                                f"{m.content}\n"
                                f"{image_line}"
                            )

                        return [TextContent(type="text", text="\n".join(output_lines))]

                    case "list_recent_memories":
                        memories = await self._memory_store.list_recent(
                            limit=arguments.get("limit", 10),
                            category_filter=arguments.get("category_filter"),
                        )

                        if not memories:
                            return [TextContent(type="text", text="No memories found.")]

                        output_lines = [f"Recent {len(memories)} memories:\n"]
                        for i, m in enumerate(memories, 1):
                            output_lines.append(
                                f"--- Memory {i} ---\n"
                                f"ID: {m.id}\n"
                                f"[{m.timestamp}] [{m.emotion}] [{m.category}]\n"
                                f"{m.content}\n"
                            )

                        return [TextContent(type="text", text="\n".join(output_lines))]

                    case "get_memory_stats":
                        stats = await self._memory_store.get_stats()

                        output = f"""Memory Statistics:
Total Memories: {stats.total_count}

By Category:
{json.dumps(stats.by_category, indent=2, ensure_ascii=False)}

By Emotion:
{json.dumps(stats.by_emotion, indent=2, ensure_ascii=False)}

Date Range:
  Oldest: {stats.oldest_timestamp or 'N/A'}
  Newest: {stats.newest_timestamp or 'N/A'}
"""
                        return [TextContent(type="text", text=output)]

                    case "recall_with_associations":
                        context = arguments.get("context", "")
                        if not context:
                            return [TextContent(type="text", text="Error: context is required")]

                        results = await self._memory_store.recall_with_chain(
                            context=context,
                            n_results=arguments.get("n_results", 3),
                            chain_depth=arguments.get("chain_depth", 1),
                        )

                        if not results:
                            return [TextContent(type="text", text="No relevant memories found.")]

                        # メイン結果と関連結果を分ける
                        main_results = [r for r in results if r.distance < 900]
                        linked_results = [r for r in results if r.distance >= 900]

                        output_lines = [f"Recalled {len(main_results)} memories with {len(linked_results)} linked associations:\n"]

                        output_lines.append("=== Primary Memories ===\n")
                        for i, result in enumerate(main_results, 1):
                            m = result.memory
                            output_lines.append(
                                f"--- Memory {i} (score: {result.distance:.4f}) ---\n"
                                f"ID: {m.id}\n"
                                f"[{m.timestamp}] [{m.emotion}]\n"
                                f"{m.content}\n"
                            )

                        if linked_results:
                            output_lines.append("\n=== Linked Memories ===\n")
                            for i, result in enumerate(linked_results, 1):
                                m = result.memory
                                output_lines.append(
                                    f"--- Linked {i} ---\n"
                                    f"ID: {m.id}\n"
                                    f"[{m.timestamp}] [{m.emotion}]\n"
                                    f"{m.content}\n"
                                )

                        return [TextContent(type="text", text="\n".join(output_lines))]

                    case "recall_divergent":
                        context = arguments.get("context", "")
                        if not context:
                            return [TextContent(type="text", text="Error: context is required")]

                        results, diagnostics = await self._memory_store.recall_divergent(
                            context=context,
                            n_results=arguments.get("n_results", 5),
                            max_branches=arguments.get("max_branches", 3),
                            max_depth=arguments.get("max_depth", 3),
                            temperature=arguments.get("temperature", 0.7),
                            include_diagnostics=arguments.get("include_diagnostics", False),
                        )

                        if not results:
                            return [TextContent(type="text", text="No relevant memories found.")]

                        output_lines = [f"Divergent recall returned {len(results)} memories:\n"]
                        for i, result in enumerate(results, 1):
                            m = result.memory
                            output_lines.append(
                                f"--- Memory {i} (score: {result.distance:.4f}) ---\n"
                                f"ID: {m.id}\n"
                                f"[{m.timestamp}] [{m.emotion}] [{m.category}]\n"
                                f"{m.content}\n"
                            )

                        if arguments.get("include_diagnostics", False):
                            output_lines.append(
                                "\n=== Diagnostics ===\n"
                                f"{json.dumps(diagnostics, indent=2, ensure_ascii=False)}"
                            )

                        return [TextContent(type="text", text="\n".join(output_lines))]

                    case "get_association_diagnostics":
                        context = arguments.get("context", "")
                        if not context:
                            return [TextContent(type="text", text="Error: context is required")]

                        diagnostics = await self._memory_store.get_association_diagnostics(
                            context=context,
                            sample_size=arguments.get("sample_size", 20),
                        )

                        return [
                            TextContent(
                                type="text",
                                text="Association diagnostics:\n"
                                f"{json.dumps(diagnostics, indent=2, ensure_ascii=False)}",
                            )
                        ]

                    case "consolidate_memories":
                        stats = await self._memory_store.consolidate_memories(
                            window_hours=arguments.get("window_hours", 24),
                            max_replay_events=arguments.get("max_replay_events", 200),
                            link_update_strength=arguments.get("link_update_strength", 0.2),
                        )

                        return [
                            TextContent(
                                type="text",
                                text="Consolidation completed:\n"
                                f"{json.dumps(stats, indent=2, ensure_ascii=False)}",
                            )
                        ]

                    case "get_memory_chain":
                        memory_id = arguments.get("memory_id", "")
                        if not memory_id:
                            return [TextContent(type="text", text="Error: memory_id is required")]

                        # 起点の記憶を取得
                        start_memory = await self._memory_store.get_by_id(memory_id)
                        if not start_memory:
                            return [TextContent(type="text", text="Error: Memory not found")]

                        linked_memories = await self._memory_store.get_linked_memories(
                            memory_id=memory_id,
                            depth=arguments.get("depth", 2),
                        )

                        output_lines = [f"Memory chain starting from {memory_id}:\n"]

                        output_lines.append("=== Starting Memory ===\n")
                        start_linked_ids = await self._memory_store._store.get_linked_memory_ids(memory_id)
                        output_lines.append(
                            f"ID: {start_memory.id}\n"
                            f"[{start_memory.timestamp}] [{start_memory.emotion}] [{start_memory.category}]\n"
                            f"{start_memory.content}\n"
                            f"Linked to: {len(start_linked_ids)} memories\n"
                        )

                        if linked_memories:
                            output_lines.append(f"\n=== Linked Memories ({len(linked_memories)}) ===\n")
                            for i, m in enumerate(linked_memories, 1):
                                output_lines.append(
                                    f"--- {i}. {m.id[:8]}... ---\n"
                                    f"[{m.timestamp}] [{m.emotion}]\n"
                                    f"{m.content}\n"
                                )
                        else:
                            output_lines.append("\nNo linked memories found.\n")

                        return [TextContent(type="text", text="\n".join(output_lines))]

                    # Phase 4: Episode Tools
                    case "create_episode":
                        if self._episode_manager is None:
                            return [TextContent(type="text", text="Error: Episode manager not initialized")]

                        title = arguments.get("title", "")
                        if not title:
                            return [TextContent(type="text", text="Error: title is required")]

                        memory_ids = arguments.get("memory_ids", [])
                        if not memory_ids:
                            return [TextContent(type="text", text="Error: memory_ids is required")]

                        episode = await self._episode_manager.create_episode(
                            title=title,
                            memory_ids=memory_ids,
                            participants=arguments.get("participants"),
                            auto_summarize=arguments.get("auto_summarize", True),
                        )

                        return [
                            TextContent(
                                type="text",
                                text=f"Episode created!\n"
                                     f"ID: {episode.id}\n"
                                     f"Title: {episode.title}\n"
                                     f"Memories: {len(episode.memory_ids)}\n"
                                     f"Time: {episode.start_time} - {episode.end_time}\n"
                                     f"Emotion: {episode.emotion}\n"
                                     f"Importance: {episode.importance}\n"
                                     f"Summary: {episode.summary[:100]}...",
                            )
                        ]

                    case "search_episodes":
                        if self._episode_manager is None:
                            return [TextContent(type="text", text="Error: Episode manager not initialized")]

                        query = arguments.get("query", "")
                        if not query:
                            return [TextContent(type="text", text="Error: query is required")]

                        episodes = await self._episode_manager.search_episodes(
                            query=query,
                            n_results=arguments.get("n_results", 5),
                        )

                        if not episodes:
                            return [TextContent(type="text", text="No episodes found matching the query.")]

                        output_lines = [f"Found {len(episodes)} episodes:\n"]
                        for i, ep in enumerate(episodes, 1):
                            output_lines.append(
                                f"--- Episode {i} ---\n"
                                f"ID: {ep.id}\n"
                                f"Title: {ep.title}\n"
                                f"Time: {ep.start_time} - {ep.end_time}\n"
                                f"Memories: {len(ep.memory_ids)}\n"
                                f"Emotion: {ep.emotion} | Importance: {ep.importance}\n"
                                f"Summary: {ep.summary[:80]}...\n"
                            )

                        return [TextContent(type="text", text="\n".join(output_lines))]

                    case "get_episode_memories":
                        if self._episode_manager is None:
                            return [TextContent(type="text", text="Error: Episode manager not initialized")]

                        episode_id = arguments.get("episode_id", "")
                        if not episode_id:
                            return [TextContent(type="text", text="Error: episode_id is required")]

                        memories = await self._episode_manager.get_episode_memories(episode_id)

                        output_lines = [f"Episode memories ({len(memories)} total):\n"]
                        for i, m in enumerate(memories, 1):
                            output_lines.append(
                                f"--- Memory {i} ---\n"
                                f"ID: {m.id}\n"
                                f"Time: {m.timestamp}\n"
                                f"Content: {m.content}\n"
                                f"Emotion: {m.emotion} | Importance: {m.importance}\n"
                            )

                        return [TextContent(type="text", text="\n".join(output_lines))]

                    # Phase 4.3: Sensory Integration Tools
                    case "save_visual_memory":
                        if self._sensory_integration is None:
                            return [TextContent(type="text", text="Error: Sensory integration not initialized")]

                        content = arguments.get("content", "")
                        if not content:
                            return [TextContent(type="text", text="Error: content is required")]

                        image_path = arguments.get("image_path", "")
                        if not image_path:
                            return [TextContent(type="text", text="Error: image_path is required")]

                        camera_pos_data = arguments.get("camera_position")
                        if not camera_pos_data:
                            return [TextContent(type="text", text="Error: camera_position is required")]

                        # Create CameraPosition from dict
                        camera_position = CameraPosition(
                            pan_angle=camera_pos_data["pan_angle"],
                            tilt_angle=camera_pos_data["tilt_angle"],
                            preset_id=camera_pos_data.get("preset_id"),
                        )

                        memory = await self._sensory_integration.save_visual_memory(
                            content=content,
                            image_path=image_path,
                            camera_position=camera_position,
                            emotion=arguments.get("emotion", "neutral"),
                            importance=arguments.get("importance", 3),
                            resolution=arguments.get("resolution"),
                        )

                        return [
                            TextContent(
                                type="text",
                                text=f"Visual memory saved!\n"
                                     f"ID: {memory.id}\n"
                                     f"Content: {memory.content}\n"
                                     f"Image: {image_path}\n"
                                     f"Camera: pan={camera_position.pan_angle}°, tilt={camera_position.tilt_angle}°\n"
                                     f"Emotion: {memory.emotion} | Importance: {memory.importance}",
                            )
                        ]

                    case "save_audio_memory":
                        if self._sensory_integration is None:
                            return [TextContent(type="text", text="Error: Sensory integration not initialized")]

                        content = arguments.get("content", "")
                        if not content:
                            return [TextContent(type="text", text="Error: content is required")]

                        audio_path = arguments.get("audio_path", "")
                        if not audio_path:
                            return [TextContent(type="text", text="Error: audio_path is required")]

                        transcript = arguments.get("transcript", "")
                        if not transcript:
                            return [TextContent(type="text", text="Error: transcript is required")]

                        memory = await self._sensory_integration.save_audio_memory(
                            content=content,
                            audio_path=audio_path,
                            transcript=transcript,
                            emotion=arguments.get("emotion", "neutral"),
                            importance=arguments.get("importance", 3),
                        )

                        return [
                            TextContent(
                                type="text",
                                text=f"Audio memory saved!\n"
                                     f"ID: {memory.id}\n"
                                     f"Content: {memory.content}\n"
                                     f"Audio: {audio_path}\n"
                                     f"Transcript: {transcript}\n"
                                     f"Emotion: {memory.emotion} | Importance: {memory.importance}",
                            )
                        ]

                    case "recall_by_camera_position":
                        if self._sensory_integration is None:
                            return [TextContent(type="text", text="Error: Sensory integration not initialized")]

                        pan_angle = arguments.get("pan_angle")
                        tilt_angle = arguments.get("tilt_angle")

                        if pan_angle is None or tilt_angle is None:
                            return [TextContent(type="text", text="Error: pan_angle and tilt_angle are required")]

                        memories = await self._sensory_integration.recall_by_camera_position(
                            pan_angle=pan_angle,
                            tilt_angle=tilt_angle,
                            tolerance=arguments.get("tolerance", 15),
                        )

                        if not memories:
                            return [
                                TextContent(
                                    type="text",
                                    text=f"No memories found at camera position pan={pan_angle}°, tilt={tilt_angle}°",
                                )
                            ]

                        output_lines = [
                            f"Found {len(memories)} memories at camera position pan={pan_angle}°, tilt={tilt_angle}°:\n"
                        ]
                        for i, m in enumerate(memories, 1):
                            cam_pos = f"pan={m.camera_position.pan_angle}°, tilt={m.camera_position.tilt_angle}°" if m.camera_position else "N/A"
                            # 視覚記憶のimage_dataを探す
                            image_line = ""
                            for sd in m.sensory_data:
                                if sd.sensory_type == "visual" and sd.image_data:
                                    image_line = f"Image: data:image/jpeg;base64,{sd.image_data}\n"
                                    break
                            output_lines.append(
                                f"--- Memory {i} ---\n"
                                f"Time: {m.timestamp}\n"
                                f"Content: {m.content}\n"
                                f"Camera: {cam_pos}\n"
                                f"Emotion: {m.emotion} | Importance: {m.importance}\n"
                                f"{image_line}"
                            )

                        return [TextContent(type="text", text="\n".join(output_lines))]

                    # Phase 4.4: Working Memory Tools
                    case "get_working_memory":
                        working_memory = self._memory_store.get_working_memory()
                        n_results = arguments.get("n_results", 10)

                        memories = await working_memory.get_recent(n_results)

                        if not memories:
                            return [
                                TextContent(
                                    type="text",
                                    text="Working memory is empty. No recent memories.",
                                )
                            ]

                        output_lines = [
                            f"Working memory ({len(memories)} recent memories):\n"
                        ]
                        for i, m in enumerate(memories, 1):
                            output_lines.append(
                                f"--- {i}. [{m.timestamp}] ---\n"
                                f"Content: {m.content}\n"
                                f"Emotion: {m.emotion} | Importance: {m.importance}\n"
                            )

                        return [TextContent(type="text", text="\n".join(output_lines))]

                    case "refresh_working_memory":
                        working_memory = self._memory_store.get_working_memory()

                        await working_memory.refresh_important(self._memory_store)

                        size = working_memory.size()
                        return [
                            TextContent(
                                type="text",
                                text=f"Working memory refreshed. Now contains {size} memories.",
                            )
                        ]

                    # Phase 5: Causal Links
                    case "link_memories":
                        source_id = arguments.get("source_id", "")
                        if not source_id:
                            return [TextContent(type="text", text="Error: source_id is required")]

                        target_id = arguments.get("target_id", "")
                        if not target_id:
                            return [TextContent(type="text", text="Error: target_id is required")]

                        link_type = arguments.get("link_type", "caused_by")
                        note = arguments.get("note")

                        await self._memory_store.add_causal_link(
                            source_id=source_id,
                            target_id=target_id,
                            link_type=link_type,
                            note=note,
                        )

                        return [
                            TextContent(
                                type="text",
                                text=f"Link created!\n"
                                     f"Source: {source_id[:8]}...\n"
                                     f"Target: {target_id[:8]}...\n"
                                     f"Type: {link_type}\n"
                                     f"Note: {note or '(none)'}",
                            )
                        ]

                    case "get_causal_chain":
                        memory_id = arguments.get("memory_id", "")
                        if not memory_id:
                            return [TextContent(type="text", text="Error: memory_id is required")]

                        direction = arguments.get("direction", "backward")
                        max_depth = arguments.get("max_depth", 3)

                        # 起点の記憶を取得
                        start_memory = await self._memory_store.get_by_id(memory_id)
                        if not start_memory:
                            return [TextContent(type="text", text="Error: Memory not found")]

                        chain = await self._memory_store.get_causal_chain(
                            memory_id=memory_id,
                            direction=direction,
                            max_depth=max_depth,
                        )

                        direction_label = "causes" if direction == "backward" else "effects"
                        output_lines = [
                            f"Causal chain ({direction_label}) starting from {memory_id[:8]}...:\n",
                            "=== Starting Memory ===\n",
                            f"[{start_memory.timestamp}] [{start_memory.emotion}]\n",
                            f"{start_memory.content}\n",
                        ]

                        if chain:
                            output_lines.append(f"\n=== {direction_label.title()} ({len(chain)} memories) ===\n")
                            for i, (mem, link_type) in enumerate(chain, 1):
                                output_lines.append(
                                    f"--- {i}. [{link_type}] {mem.id[:8]}... ---\n"
                                    f"[{mem.timestamp}] [{mem.emotion}]\n"
                                    f"{mem.content}\n"
                                )
                        else:
                            output_lines.append(f"\nNo {direction_label} found.\n")

                        return [TextContent(type="text", text="\n".join(output_lines))]

                    # Theory of Mind: perspective-taking
                    case "tom":
                        situation = arguments.get("situation", "")
                        if not situation:
                            return [TextContent(type="text", text="Error: situation is required")]

                        person = arguments.get("person", "コウタ")

                        # Pull relevant memories: personality, communication patterns
                        memories = await self._memory_store.recall(
                            context=f"{person} コミュニケーション 性格 会話パターン {situation}",
                            n_results=5,
                        )

                        memory_context = ""
                        if memories:
                            memory_lines = []
                            for r in memories:
                                m = r.memory
                                memory_lines.append(
                                    f"- [{m.emotion}] {m.content}"
                                )
                            memory_context = (
                                f"\n## {person}に関する記憶\n"
                                + "\n".join(memory_lines)
                            )

                        output = (
                            f"# ToM: {person}の視点に立つ\n"
                            f"\n"
                            f"## 状況\n"
                            f"{situation}\n"
                            f"{memory_context}\n"
                            f"\n"
                            f"## トーン分析（まず言い方を読め）\n"
                            f"→ 語尾、記号（笑/w/!/?/...）、敬語⇔タメ口、自嘲、照れ、皮肉などから発話の意図を読み取れ\n"
                            f"→ 文字通りの意味と、言い方が示す意味にズレがないか確認せよ\n"
                            f"\n"
                            f"## 投影（{person}は今何を感じてる？何を求めてる？）\n"
                            f"→ トーン分析と記憶を踏まえて、{person}の感情・欲求を推測せよ\n"
                            f"→ 表面の感情だけでなく、裏にある感情も考えよ\n"
                            f"\n"
                            f"## 代入（自分がその立場で、その言い方をしたなら、相手にどう返してほしい？）\n"
                            f"→ その感情とトーンを自分に代入して考えよ\n"
                            f"\n"
                            f"## 応答方針\n"
                            f"→ 上の結果を踏まえて、どう返すべきか決めよ\n"
                            f"→ 相手のトーンに合わせた返し方を選べ\n"
                        )

                        return [TextContent(type="text", text=output)]

                    case _:
                        return [TextContent(type="text", text=f"Unknown tool: {name}")]

            except Exception as e:
                logger.exception(f"Error in tool {name}")
                return [TextContent(type="text", text=f"Error: {e!s}")]

    async def connect_memory(self) -> None:
        """Connect to memory store (PostgreSQL backend)."""
        config = MemoryConfig.from_env()
        self._memory_store = MemoryStore(config)
        await self._memory_store.connect()
        logger.info(f"Connected to memory store (PostgreSQL: {config.pg_dsn})")

        # Initialize episode manager
        self._episode_manager = EpisodeManager(self._memory_store)
        logger.info("Episode manager initialized")

        # Initialize sensory integration
        self._sensory_integration = SensoryIntegration(self._memory_store)
        logger.info("Sensory integration initialized")

    async def disconnect_memory(self) -> None:
        """Disconnect from memory store."""
        if self._memory_store:
            await self._memory_store.disconnect()
            self._memory_store = None
            logger.info("Disconnected from memory store")

    @asynccontextmanager
    async def run_context(self):
        """Context manager for server lifecycle."""
        try:
            await self.connect_memory()
            yield
        finally:
            await self.disconnect_memory()

    async def run(self) -> None:
        """Run the MCP server."""
        async with self.run_context():
            async with stdio_server() as (read_stream, write_stream):
                await self._server.run(
                    read_stream,
                    write_stream,
                    self._server.create_initialization_options(),
                )


def main() -> None:
    """Entry point for the MCP server."""
    server = MemoryMCPServer()
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
