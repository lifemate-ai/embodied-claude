"""DDL definitions for PostgreSQL schema."""

# Schema version for future migrations
SCHEMA_VERSION = 1

SETUP_EXTENSIONS = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgroonga;
"""

CREATE_EPISODES_TABLE = """
CREATE TABLE IF NOT EXISTS episodes (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title           TEXT NOT NULL,
    start_time      TIMESTAMPTZ NOT NULL,
    end_time        TIMESTAMPTZ,
    participants    TEXT[] NOT NULL DEFAULT '{{}}',
    location_context TEXT,
    summary         TEXT NOT NULL,
    embedding       vector({dim}),
    emotion         TEXT NOT NULL DEFAULT 'neutral',
    importance      SMALLINT NOT NULL DEFAULT 3
);
"""

CREATE_MEMORIES_TABLE = """
CREATE TABLE IF NOT EXISTS memories (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content          TEXT NOT NULL,
    embedding        vector({dim}) NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    emotion          TEXT NOT NULL DEFAULT 'neutral'
                     CHECK (emotion IN (
                         'happy','sad','surprised','moved',
                         'excited','nostalgic','curious','neutral'
                     )),
    importance       SMALLINT NOT NULL DEFAULT 3
                     CHECK (importance BETWEEN 1 AND 5),
    category         TEXT NOT NULL DEFAULT 'daily'
                     CHECK (category IN (
                         'daily','philosophical','technical',
                         'memory','observation','feeling','conversation'
                     )),
    access_count     INTEGER NOT NULL DEFAULT 0,
    last_accessed    TIMESTAMPTZ,
    episode_id       UUID REFERENCES episodes(id) ON DELETE SET NULL,
    sensory_data     JSONB NOT NULL DEFAULT '[]'::jsonb,
    camera_position  JSONB,
    tags             TEXT[] NOT NULL DEFAULT '{{}}',
    novelty_score    REAL NOT NULL DEFAULT 0.0,
    prediction_error REAL NOT NULL DEFAULT 0.0,
    activation_count INTEGER NOT NULL DEFAULT 0,
    last_activated   TIMESTAMPTZ
);
"""

CREATE_MEMORY_LINKS_TABLE = """
CREATE TABLE IF NOT EXISTS memory_links (
    source_id   UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    target_id   UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    link_type   TEXT NOT NULL DEFAULT 'related'
                CHECK (link_type IN ('similar','caused_by','leads_to','related')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    note        TEXT,
    PRIMARY KEY (source_id, target_id, link_type)
);
"""

CREATE_COACTIVATION_TABLE = """
CREATE TABLE IF NOT EXISTS coactivation_weights (
    memory_a   UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    memory_b   UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    weight     REAL NOT NULL DEFAULT 0.0 CHECK (weight BETWEEN 0.0 AND 1.0),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (memory_a, memory_b),
    CHECK (memory_a < memory_b)
);
"""

CREATE_INDEXES = """
-- Vector nearest-neighbor (HNSW)
CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);

-- pgroonga full-text search (MeCab tokenizer + NFKC normalization)
CREATE INDEX IF NOT EXISTS idx_memories_content_pgroonga ON memories
    USING pgroonga (content)
    WITH (tokenizer='TokenMecab', normalizer='NormalizerNFKC150');

-- Metadata filters
CREATE INDEX IF NOT EXISTS idx_memories_emotion ON memories (emotion);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories (importance);
CREATE INDEX IF NOT EXISTS idx_memories_category ON memories (category);
CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_episode_id ON memories (episode_id) WHERE episode_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories USING GIN (tags);
CREATE INDEX IF NOT EXISTS idx_memories_sensory_data ON memories USING GIN (sensory_data);

-- Working memory refresh (composite)
CREATE INDEX IF NOT EXISTS idx_memories_refresh ON memories (importance, access_count, created_at DESC)
    WHERE importance >= 4 AND access_count >= 5;

-- Link table (graph traversal)
CREATE INDEX IF NOT EXISTS idx_links_source ON memory_links (source_id);
CREATE INDEX IF NOT EXISTS idx_links_target ON memory_links (target_id);
CREATE INDEX IF NOT EXISTS idx_links_type ON memory_links (link_type);

-- Coactivation weights
CREATE INDEX IF NOT EXISTS idx_coactivation_a ON coactivation_weights (memory_a);
CREATE INDEX IF NOT EXISTS idx_coactivation_b ON coactivation_weights (memory_b);
CREATE INDEX IF NOT EXISTS idx_coactivation_weight ON coactivation_weights (weight DESC)
    WHERE weight >= 0.3;

-- Episodes
CREATE INDEX IF NOT EXISTS idx_episodes_embedding ON episodes
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);
CREATE INDEX IF NOT EXISTS idx_episodes_time ON episodes (start_time DESC);
"""


def get_all_ddl(dim: int = 768) -> list[str]:
    """Return all DDL statements in dependency order."""
    return [
        SETUP_EXTENSIONS,
        CREATE_EPISODES_TABLE.format(dim=dim),
        CREATE_MEMORIES_TABLE.format(dim=dim),
        CREATE_MEMORY_LINKS_TABLE,
        CREATE_COACTIVATION_TABLE,
        CREATE_INDEXES,
    ]
