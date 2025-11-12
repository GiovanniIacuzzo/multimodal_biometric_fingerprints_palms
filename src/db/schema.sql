-- ===========================
--  SUBJECTS
-- ===========================
CREATE TABLE IF NOT EXISTS subjects (
    id SERIAL PRIMARY KEY,
    subject_code VARCHAR(50) UNIQUE NOT NULL,
    notes TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- ===========================
--  IMAGES
-- ===========================
CREATE TABLE IF NOT EXISTS images (
    id SERIAL PRIMARY KEY,
    subject_id INTEGER REFERENCES subjects(id) ON DELETE CASCADE,
    filename VARCHAR(255) NOT NULL,
    path_original TEXT NOT NULL,
    path_enhanced TEXT,
    path_skeleton TEXT,
    orientation_mean FLOAT,
    preprocessing_time_sec FLOAT,
    status VARCHAR(30),
    cluster_id INTEGER REFERENCES clusters(id) ON DELETE SET NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- ===========================
--  CLUSTERS
-- ===========================
CREATE TABLE IF NOT EXISTS clusters (
    id SERIAL PRIMARY KEY,
    cluster_label VARCHAR(50),
    algorithm VARCHAR(50),
    n_members INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- ===========================
--  MINUTIAE
-- ===========================
CREATE TABLE IF NOT EXISTS minutiae (
    id SERIAL PRIMARY KEY,
    image_id INTEGER REFERENCES images(id) ON DELETE CASCADE,
    x FLOAT NOT NULL,
    y FLOAT NOT NULL,
    type VARCHAR(20),
    orientation FLOAT,
    quality FLOAT,
    coherence FLOAT,
    cn SMALLINT,
    deg SMALLINT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- ===========================
--  FEATURES SUMMARY
-- ===========================
CREATE TABLE IF NOT EXISTS features_summary (
    id SERIAL PRIMARY KEY,
    image_id INTEGER REFERENCES images(id) ON DELETE CASCADE,
    raw_count INTEGER,
    post_count INTEGER,
    avg_quality FLOAT,
    avg_coherence FLOAT,
    processing_time_sec FLOAT,
    params_json JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- ===========================
--  MATCHING RESULTS
-- ===========================
CREATE TABLE IF NOT EXISTS matching_results (
    id SERIAL PRIMARY KEY,
    image_a_id INTEGER REFERENCES images(id) ON DELETE CASCADE,
    image_b_id INTEGER REFERENCES images(id) ON DELETE CASCADE,
    score FLOAT,
    method VARCHAR(50),
    same_cluster BOOLEAN,
    created_at TIMESTAMP DEFAULT NOW()
);

-- ===========================
--  PIPELINE RUNS / EXPERIMENTS
-- ===========================
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    description TEXT,
    algorithm VARCHAR(50),
    params_json JSONB,
    execution_time_sec FLOAT,
    n_clusters INTEGER,
    avg_silhouette FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- ===========================
--  CLUSTER MEMBERSHIPS
-- ===========================
CREATE TABLE IF NOT EXISTS cluster_memberships (
    id SERIAL PRIMARY KEY,
    cluster_id INTEGER REFERENCES clusters(id) ON DELETE CASCADE,
    image_id INTEGER REFERENCES images(id) ON DELETE CASCADE,
    distance FLOAT,
    assigned_in_run INTEGER REFERENCES pipeline_runs(id),
    created_at TIMESTAMP DEFAULT NOW()
);
