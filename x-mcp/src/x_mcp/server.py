"""grok-mcp — MCP server for searching X (Twitter) via xAI Grok live search."""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

_API_KEY = os.environ.get("GROK_API_KEY", "") or os.environ.get("XAI_API_KEY", "")
_BASE_URL = "https://api.x.ai/v1"

mcp = FastMCP("grok-mcp")


def _client() -> OpenAI:
    if not _API_KEY:
        raise RuntimeError("XAI_API_KEY is not set")
    return OpenAI(api_key=_API_KEY, base_url=_BASE_URL)


def _x_search(query: str, max_results: int = 10) -> str:
    """Call Grok Responses API with x_search tool enabled."""
    client = _client()
    response = client.responses.create(
        model="grok-4",
        tools=[
            {
                "type": "x_search",
            }
        ],
        input=query,
        max_output_tokens=2048,
    )
    return response.output_text or "(no results)"


@mcp.tool()
def search_x(query: str, count: int = 10) -> str:
    """Search X (Twitter) posts in real-time using Grok live search.

    Args:
        query: Search query. Supports keywords, hashtags (#familiar_ai),
               @mentions, and natural language questions.
        count: Approximate number of results to aim for (1-20).
    """
    prompt = f"Search X for: {query}\nReturn the {count} most relevant recent posts. For each post include: username, post text, approximate time, and engagement (likes/reposts if available). Format as a clear list."
    return _x_search(prompt, count)


@mcp.tool()
def get_user_tweets(username: str, count: int = 10) -> str:
    """Get recent tweets from a specific X user.

    Args:
        username: X username without @ (e.g. 'kmizu')
        count: Number of recent tweets to fetch (1-20).
    """
    prompt = f"Search X for recent posts from @{username}. Return the {count} most recent posts with text, timestamp, and engagement stats."
    return _x_search(prompt, count)


@mcp.tool()
def get_mentions(username: str, count: int = 10) -> str:
    """Get recent mentions and replies to a specific X user.

    Args:
        username: X username without @ (e.g. 'kmizu')
        count: Number of recent mentions to fetch (1-20).
    """
    prompt = f"Search X for recent posts that mention or reply to @{username}. Return the {count} most recent mentions with username, post text, and timestamp."
    return _x_search(prompt, count)


@mcp.tool()
def get_trending_topic(topic: str) -> str:
    """Get a summary of what's being said about a topic or hashtag on X right now.

    Args:
        topic: Topic, hashtag, or keyword to summarize (e.g. '#familiar_ai', 'Claude Code')
    """
    prompt = f"Search X for '{topic}' and give me a summary of what people are saying right now. Include key opinions, notable posts, and the overall sentiment. Also list 3-5 representative quotes with usernames."
    return _x_search(prompt)


# ── X/Twitter Posting ─────────────────────────────────────────

_X_CONSUMER_KEY = os.environ.get("X_CONSUMER_KEY", "")
_X_CONSUMER_SECRET = os.environ.get("X_CONSUMER_SECRET", "")
_X_ACCESS_TOKEN = os.environ.get("X_ACCESS_TOKEN", "")
_X_ACCESS_TOKEN_SECRET = os.environ.get("X_ACCESS_TOKEN_SECRET", "")


def _tweepy_client():
    """Create tweepy Client (v2) for posting tweets."""
    import tweepy

    if not all([_X_CONSUMER_KEY, _X_CONSUMER_SECRET, _X_ACCESS_TOKEN, _X_ACCESS_TOKEN_SECRET]):
        raise RuntimeError(
            "X_CONSUMER_KEY, X_CONSUMER_SECRET, X_ACCESS_TOKEN, and X_ACCESS_TOKEN_SECRET must be set"
        )
    return tweepy.Client(
        consumer_key=_X_CONSUMER_KEY,
        consumer_secret=_X_CONSUMER_SECRET,
        access_token=_X_ACCESS_TOKEN,
        access_token_secret=_X_ACCESS_TOKEN_SECRET,
    )


def _tweepy_api():
    """Create tweepy API (v1.1) for media upload."""
    import tweepy

    auth = tweepy.OAuth1UserHandler(
        _X_CONSUMER_KEY, _X_CONSUMER_SECRET,
        _X_ACCESS_TOKEN, _X_ACCESS_TOKEN_SECRET,
    )
    return tweepy.API(auth)


@mcp.tool()
def post_tweet(text: str, image_path: str = "", reply_to: str = "") -> str:
    """Post a tweet to X as @xai_kokone, optionally with an image.

    IMPORTANT: X uses weighted character count. Japanese/CJK characters count as 2.
    Effective limit is ~140 Japanese characters (= 280 weighted).
    Keep Japanese tweets under 140 chars to be safe.

    Args:
        text: Tweet text (max 280 weighted characters; ~140 Japanese chars).
        image_path: Path to image file to attach (optional, jpg/png).
        reply_to: Tweet ID to reply to (optional, leave empty for new tweet).
    """
    # X weighted character count: CJK/Japanese = 2, ASCII = 1
    weighted = sum(2 if ord(c) > 0x7F else 1 for c in text)
    if weighted > 280:
        return f"Error: Tweet is {weighted} weighted characters (max 280). Japanese chars count as 2. Shorten to ~{280 - (weighted - 280) // 2} chars."

    client = _tweepy_client()
    kwargs = {"text": text}

    # Upload image if provided
    if image_path:
        api = _tweepy_api()
        media = api.media_upload(filename=image_path)
        kwargs["media_ids"] = [media.media_id]

    if reply_to:
        kwargs["in_reply_to_tweet_id"] = reply_to

    response = client.create_tweet(**kwargs)
    tweet_id = response.data["id"]
    return f"Posted! https://x.com/xai_kokone/status/{tweet_id}"


@mcp.tool()
def delete_tweet(tweet_id: str) -> str:
    """Delete a tweet by ID.

    Args:
        tweet_id: The tweet ID to delete.
    """
    client = _tweepy_client()
    client.delete_tweet(tweet_id)
    return f"Deleted tweet {tweet_id}"


def main():
    mcp.run()


if __name__ == "__main__":
    main()
