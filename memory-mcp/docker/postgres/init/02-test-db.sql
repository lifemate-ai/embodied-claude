-- Create test database for running tests
CREATE DATABASE embodied_claude_test OWNER memory_mcp;

-- Enable extensions in test database
\c embodied_claude_test
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgroonga;
