-- Database initialization script for ARKOS
-- This script is run when the PostgreSQL container starts

-- Create conversation_context table for short-term memory
CREATE TABLE IF NOT EXISTS conversation_context (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL,
    message TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_conversation_user_id ON conversation_context(user_id);
CREATE INDEX IF NOT EXISTS idx_conversation_session_id ON conversation_context(session_id);
CREATE INDEX IF NOT EXISTS idx_conversation_created_at ON conversation_context(created_at);

-- Composite index for common query pattern
CREATE INDEX IF NOT EXISTS idx_conversation_user_session ON conversation_context(user_id, session_id, created_at DESC);
