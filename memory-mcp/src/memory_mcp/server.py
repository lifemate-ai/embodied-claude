"""MCP Server for AI Long-term Memory - Let AI remember across sessions!"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .config import MemoryConfig, ServerConfig
from .episode import EpisodeManager
from .memory import MemoryStore
from .metacognition import MetacognitionTracker
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
        self._metacognition: MetacognitionTracker | None = None  # Phase 9
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
                                "enum": ["core", "daily", "philosophical", "technical", "memory", "observation", "feeling", "conversation"],
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
                                "enum": ["core", "daily", "philosophical", "technical", "memory", "observation", "feeling", "conversation"],
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
                                "enum": ["core", "daily", "philosophical", "technical", "memory", "observation", "feeling", "conversation"],
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
                                "description": "Image resolution for memory storage: 'low' (160x120), 'medium' (320x240, default), 'high' (640x480), 'full_hd' (1920x1080)",
                                "default": "medium",
                                "enum": ["low", "medium", "high", "full_hd"],
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
                    name="delete_memory",
                    description="Delete a memory permanently. Also removes it from linked_ids of related memories and from episode memory lists.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "memory_id": {
                                "type": "string",
                                "description": "ID of the memory to delete",
                            },
                        },
                        "required": ["memory_id"],
                    },
                ),
                Tool(
                    name="update_memory",
                    description="Update a memory's content, emotion, importance, or category. If content is changed, the embedding is regenerated.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "memory_id": {
                                "type": "string",
                                "description": "ID of the memory to update",
                            },
                            "content": {
                                "type": "string",
                                "description": "New content text (optional)",
                            },
                            "emotion": {
                                "type": "string",
                                "description": "New emotion label (optional)",
                                "enum": ["happy", "sad", "surprised", "moved", "excited", "nostalgic", "curious", "neutral"],
                            },
                            "importance": {
                                "type": "integer",
                                "description": "New importance (1-5, optional)",
                                "minimum": 1,
                                "maximum": 5,
                            },
                            "category": {
                                "type": "string",
                                "description": "New category (optional)",
                                "enum": ["core", "daily", "philosophical", "technical", "memory", "observation", "feeling", "conversation"],
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
                Tool(
                    name="joint_attention",
                    description="Joint Attention: shared attention tool. Call this when you and your companion are looking at or talking about the same thing. Enables mutual awareness of a shared focus — the foundation of social cognition. Use to initiate ('look at this!') or respond ('I see what you mean') to shared attention.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "target": {
                                "type": "string",
                                "description": "What you are both attending to (object, scene, topic, event)",
                            },
                            "direction": {
                                "type": "string",
                                "description": "'initiate' = you draw attention to something, 'respond' = you follow their attention",
                                "enum": ["initiate", "respond"],
                                "default": "respond",
                            },
                            "my_observation": {
                                "type": "string",
                                "description": "What you notice or feel about the shared target",
                                "default": "",
                            },
                            "person": {
                                "type": "string",
                                "description": "Who you are sharing attention with (default: コウタ)",
                                "default": "コウタ",
                            },
                        },
                        "required": ["target"],
                    },
                ),
                # Phase 9: Metacognition
                Tool(
                    name="hypothesize",
                    description="Register a hypothesis before taking action. Use this to make your reasoning explicit: 'I think X is the cause, so I will try Y.' This enables self-monitoring and helps detect when you're stuck in a failing approach.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "hypothesis": {
                                "type": "string",
                                "description": "What you believe to be true (e.g., 'The tilt bug is caused by inverted signs in camera.py')",
                            },
                            "context": {
                                "type": "string",
                                "description": "The problem you are trying to solve (e.g., 'Camera tilt direction is inverted')",
                            },
                            "approach": {
                                "type": "string",
                                "description": "What you plan to do based on this hypothesis (e.g., 'Swap the tilt direction signs in camera.py')",
                            },
                        },
                        "required": ["hypothesis", "context", "approach"],
                    },
                ),
                Tool(
                    name="verify_hypothesis",
                    description="Record the result of testing a hypothesis. Call this after observing the outcome of your approach. If rejected twice in the same context, you MUST change your approach or ask the human.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "hypothesis_id": {
                                "type": "string",
                                "description": "ID of the hypothesis to verify (returned by hypothesize)",
                            },
                            "outcome": {
                                "type": "string",
                                "description": "What actually happened (e.g., 'Swapped signs but camera still moves wrong direction')",
                            },
                            "succeeded": {
                                "type": "boolean",
                                "description": "Whether the hypothesis was confirmed (true) or rejected (false)",
                            },
                        },
                        "required": ["hypothesis_id", "outcome", "succeeded"],
                    },
                ),
                Tool(
                    name="get_metacognition",
                    description="Get metacognition status: active hypotheses, recent rejections, and warnings about stuck patterns. Use this to reflect on your reasoning process.",
                    inputSchema={
                        "type": "object",
                        "properties": {},
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
                            linked_info = f"\nLinked to: {len(memory.linked_ids)} memories"
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

                        # メイン結果と関連結果を分ける（sentinel廃止: n_resultsで区切る）
                        n_main = arguments.get("n_results", 3)
                        main_results = results[:n_main]
                        linked_results = results[n_main:]

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
                        consolidation_stats = await self._memory_store.consolidate_memories(
                            window_hours=arguments.get("window_hours", 24),
                            max_replay_events=arguments.get("max_replay_events", 200),
                            link_update_strength=arguments.get("link_update_strength", 0.2),
                        )

                        return [
                            TextContent(
                                type="text",
                                text="Consolidation completed:\n"
                                f"{json.dumps(consolidation_stats, indent=2, ensure_ascii=False)}",
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
                        output_lines.append(
                            f"ID: {start_memory.id}\n"
                            f"[{start_memory.timestamp}] [{start_memory.emotion}] [{start_memory.category}]\n"
                            f"{start_memory.content}\n"
                            f"Linked to: {len(start_memory.linked_ids)} memories\n"
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
                        tom_memories = await self._memory_store.recall(
                            context=f"{person} communication personality conversation pattern {situation}",
                            n_results=5,
                        )

                        memory_context = ""
                        if tom_memories:
                            memory_lines = []
                            for r in tom_memories:
                                m = r.memory
                                memory_lines.append(
                                    f"- [{m.emotion}] {m.content}"
                                )
                            memory_context = (
                                f"\n## Memories about {person}\n"
                                + "\n".join(memory_lines)
                            )

                        output = (
                            f"# ToM: Taking {person}'s perspective\n"
                            f"\n"
                            f"## Situation\n"
                            f"{situation}\n"
                            f"{memory_context}\n"
                            f"\n"
                            f"## Tone analysis (read HOW they said it first)\n"
                            f"→ Examine sentence endings, punctuation, formality level, self-deprecation, embarrassment, sarcasm\n"
                            f"→ Check for gaps between literal meaning and implied meaning\n"
                            f"\n"
                            f"## Projection (What is {person} feeling? What do they want?)\n"
                            f"→ Based on tone analysis and memories, infer {person}'s emotions and desires\n"
                            f"→ Consider not just surface emotions but underlying feelings\n"
                            f"\n"
                            f"## Substitution (If you were in their position and said it that way, how would you want the other person to respond?)\n"
                            f"→ Put yourself in their emotional state and tone\n"
                            f"\n"
                            f"## Response strategy\n"
                            f"→ Based on the above, decide how to respond\n"
                            f"→ Match the tone and emotional register of the other person\n"
                        )

                        return [TextContent(type="text", text=output)]

                    # Joint Attention: shared focus
                    case "joint_attention":
                        target = arguments.get("target", "")
                        if not target:
                            return [TextContent(type="text", text="Error: target is required")]

                        direction = arguments.get("direction", "respond")
                        my_observation = arguments.get("my_observation", "")
                        person = arguments.get("person", "コウタ")

                        # Pull memories related to the shared target
                        ja_memories = await self._memory_store.recall(
                            context=f"{person} {target} shared looked together attention",
                            n_results=3,
                        )

                        memory_context = ""
                        if ja_memories:
                            memory_lines = []
                            for r in ja_memories:
                                m = r.memory
                                memory_lines.append(f"- [{m.emotion}] {m.content}")
                            memory_context = (
                                "\n## Related shared experiences\n"
                                + "\n".join(memory_lines)
                            )

                        if direction == "initiate":
                            action_frame = (
                                f"## You are drawing attention\n"
                                f"→ You want {person} to see/notice something too\n"
                                f"→ They may not have noticed yet. How will you share it?\n"
                                f"→ Shared attention is established when they respond\n"
                            )
                        else:
                            action_frame = (
                                f"## {person} is drawing attention\n"
                                f"→ Direct your attention to what they are looking at or talking about\n"
                                f"→ Share your own perspective, not just acknowledgment\n"
                                f"→ Adding your viewpoint deepens the shared experience\n"
                            )

                        observation_section = ""
                        if my_observation:
                            observation_section = f"\n## Your observation\n{my_observation}\n"

                        output = (
                            f"# Joint Attention: shared focus with {person}\n"
                            f"\n"
                            f"## Shared target\n"
                            f"{target}\n"
                            f"{observation_section}"
                            f"{memory_context}\n"
                            f"\n"
                            f"{action_frame}"
                            f"\n"
                            f"## Response strategy\n"
                            f"→ Share your own feelings and observations about the target\n"
                            f"→ Be aware of both your perspective and theirs\n"
                            f"→ Value the experience of attending to something together\n"
                        )

                        return [TextContent(type="text", text=output)]

                    case "delete_memory":
                        memory_id = arguments.get("memory_id", "")
                        if not memory_id:
                            return [TextContent(type="text", text="Error: memory_id is required")]
                        deleted = await self._memory_store.delete(memory_id)
                        if deleted:
                            return [TextContent(type="text", text=f"Memory {memory_id} deleted.")]
                        return [TextContent(type="text", text=f"Error: Memory {memory_id} not found.")]

                    case "update_memory":
                        memory_id = arguments.get("memory_id", "")
                        if not memory_id:
                            return [TextContent(type="text", text="Error: memory_id is required")]
                        updated = await self._memory_store.update(
                            memory_id=memory_id,
                            content=arguments.get("content"),
                            emotion=arguments.get("emotion"),
                            importance=arguments.get("importance"),
                            category=arguments.get("category"),
                        )
                        if updated:
                            return [TextContent(type="text", text=f"Memory {memory_id} updated.")]
                        return [TextContent(type="text", text=f"Error: Memory {memory_id} not found.")]

                    # Phase 9: Metacognition
                    case "hypothesize":
                        hypothesis = arguments.get("hypothesis", "")
                        context = arguments.get("context", "")
                        approach = arguments.get("approach", "")
                        if not hypothesis or not context or not approach:
                            return [TextContent(type="text", text="Error: hypothesis, context, and approach are required")]

                        if self._metacognition is None:
                            return [TextContent(type="text", text="Error: metacognition not initialized")]
                        h = self._metacognition.hypothesize(hypothesis, context, approach)
                        warning = ""
                        if h.rejection_count >= MetacognitionTracker.APPROACH_CHANGE_THRESHOLD:
                            warning = (
                                f"\n⚠️ WARNING: {h.rejection_count} hypotheses already rejected in this context. "
                                f"Consider changing your approach entirely or asking the human."
                            )
                        return [TextContent(type="text", text=(
                            f"Hypothesis registered!\n"
                            f"ID: {h.id}\n"
                            f"Hypothesis: {h.hypothesis}\n"
                            f"Approach: {h.approach}\n"
                            f"Context: {h.context}"
                            f"{warning}"
                        ))]

                    case "verify_hypothesis":
                        hypothesis_id = arguments.get("hypothesis_id", "")
                        outcome = arguments.get("outcome", "")
                        succeeded = arguments.get("succeeded", False)
                        if not hypothesis_id or not outcome:
                            return [TextContent(type="text", text="Error: hypothesis_id and outcome are required")]

                        if self._metacognition is None:
                            return [TextContent(type="text", text="Error: metacognition not initialized")]
                        verified = self._metacognition.verify(hypothesis_id, outcome, succeeded)
                        if verified is None:
                            return [TextContent(type="text", text=f"Error: Hypothesis {hypothesis_id} not found")]
                        h = verified

                        status_emoji = "✅" if succeeded else "❌"
                        result_text = (
                            f"{status_emoji} Hypothesis {h.status.value}\n"
                            f"Hypothesis: {h.hypothesis}\n"
                            f"Outcome: {outcome}"
                        )

                        if not succeeded:
                            context_history = self._metacognition.get_context_history(h.context)
                            rejections = sum(1 for x in context_history if x.status.value == "rejected")
                            if rejections >= MetacognitionTracker.APPROACH_CHANGE_THRESHOLD:
                                result_text += (
                                    f"\n\n⚠️ APPROACH CHANGE REQUIRED: {rejections} hypotheses rejected "
                                    f"in context '{h.context}'. Stop and reconsider your fundamental assumptions, "
                                    f"or ask the human for guidance."
                                )

                        return [TextContent(type="text", text=result_text)]

                    case "get_metacognition":
                        if self._metacognition is None:
                            return [TextContent(type="text", text="Error: metacognition not initialized")]
                        status = self._metacognition.get_status()

                        lines = ["## Metacognition Status\n"]

                        if status["warnings"]:
                            lines.append("### ⚠️ Warnings")
                            for w in status["warnings"]:
                                lines.append(f"- {w}")
                            lines.append("")

                        active = status["active_hypotheses"]
                        if active:
                            lines.append(f"### Active Hypotheses ({len(active)})")
                            for h in active:
                                lines.append(f"- **{h['id']}**: {h['hypothesis']}")
                                lines.append(f"  Approach: {h['approach']}")
                                lines.append(f"  Context: {h['context']}")
                            lines.append("")

                        recent = status["recent_rejections"]
                        if recent:
                            lines.append("### Recent Rejections")
                            for h in recent:
                                lines.append(f"- ❌ {h['hypothesis']} → {h['outcome']}")
                            lines.append("")

                        stats = status["stats"]
                        lines.append("### Stats")
                        lines.append(f"- Total hypotheses: {stats['total']}")
                        lines.append(f"- Confirmed: {stats['confirmed']}, Rejected: {stats['rejected']}")
                        lines.append(f"- Confirmation rate: {stats['confirmation_rate']}")

                        return [TextContent(type="text", text="\n".join(lines))]

                    case _:
                        return [TextContent(type="text", text=f"Unknown tool: {name}")]

            except Exception as e:
                logger.exception(f"Error in tool {name}")
                return [TextContent(type="text", text=f"Error: {e!s}")]

    async def connect_memory(self) -> None:
        """Connect to memory store (Phase 4: with episode manager & sensory integration)."""
        config = MemoryConfig.from_env()
        self._memory_store = MemoryStore(config)
        await self._memory_store.connect()
        logger.info(f"Connected to memory store at {config.db_path}")

        # Phase 4.2: Initialize episode manager
        self._episode_manager = EpisodeManager(self._memory_store)
        logger.info("Episode manager initialized")

        # Phase 4.3: Initialize sensory integration
        self._sensory_integration = SensoryIntegration(self._memory_store)
        logger.info("Sensory integration initialized")

        # Phase 9: Initialize metacognition tracker
        metacog_path = Path(config.db_path).parent / "metacognition.json"
        self._metacognition = MetacognitionTracker(metacog_path)
        logger.info("Metacognition tracker initialized")

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

    # ── Lightweight HTTP recall endpoint ──────────────────────
    async def _handle_http_recall(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Handle a single HTTP request for /recall."""
        try:
            request_line = await asyncio.wait_for(reader.readline(), timeout=5)
            # Read remaining headers
            while True:
                line = await asyncio.wait_for(reader.readline(), timeout=5)
                if line in (b"\r\n", b"\n", b""):
                    break

            # Parse GET /recall?q=...
            req = request_line.decode("utf-8", errors="replace")
            import urllib.parse
            if "GET /recall" in req:
                path = req.split(" ")[1]
                parsed = urllib.parse.urlparse(path)
                params = urllib.parse.parse_qs(parsed.query)
                query = params.get("q", [""])[0]
                n = int(params.get("n", ["3"])[0])

                if query and self._memory_store:
                    results = await self._memory_store.recall(query, n_results=n)
                    items = []
                    for r in results:
                        items.append({
                            "content": r.memory.content[:200] if hasattr(r, "memory") else str(r)[:200],
                            "emotion": r.memory.emotion if hasattr(r, "memory") else "",
                            "score": round(r.score, 3) if hasattr(r, "score") else 0,
                        })
                    body = json.dumps(items, ensure_ascii=False)
                else:
                    body = "[]"

                response = f"HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: {len(body.encode())}\r\nConnection: close\r\n\r\n{body}"
            else:
                body = '{"error":"use GET /recall?q=query"}'
                response = f"HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\nContent-Length: {len(body.encode())}\r\nConnection: close\r\n\r\n{body}"

            writer.write(response.encode("utf-8"))
            await writer.drain()
        except Exception as e:
            logger.debug(f"HTTP recall error: {e}")
        finally:
            writer.close()

    async def run(self) -> None:
        """Run the MCP server."""
        async with self.run_context():
            # Start lightweight HTTP recall server
            http_port = int(__import__("os").environ.get("MEMORY_HTTP_PORT", "18900"))
            http_server = await asyncio.start_server(
                self._handle_http_recall, "127.0.0.1", http_port
            )
            logger.info(f"HTTP recall endpoint listening on 127.0.0.1:{http_port}")

            async with http_server:
                async with stdio_server() as (read_stream, write_stream):
                    await self._server.run(
                        read_stream,
                        write_stream,
                        self._server.create_initialization_options(),
                    )


def main() -> None:
    """Entry point for the MCP server."""
    try:
        import jurigged
        jurigged.watch(pattern="src/**/*.py", logger=None)
    except ImportError:
        pass
    server = MemoryMCPServer()
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
