CREATE TABLE IF NOT EXISTS subjects (
    id SERIAL PRIMARY KEY,
    subject_code VARCHAR(50) UNIQUE NOT NULL,
    notes TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS images (
    id SERIAL PRIMARY KEY,
    subject_id INTEGER REFERENCES subjects(id) ON DELETE CASCADE,
    filename VARCHAR(255),
    path_original TEXT,
    path_enhanced TEXT,
    path_skeleton TEXT,
    orientation_mean FLOAT,
    preprocessing_time_sec FLOAT,
    status VARCHAR(30),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS minutiae (
    id SERIAL PRIMARY KEY,
    image_id INTEGER REFERENCES images(id) ON DELETE CASCADE,
    x FLOAT,
    y FLOAT,
    type VARCHAR(20),
    orientation FLOAT,
    quality FLOAT,
    coherence FLOAT,
    cn SMALLINT,
    deg SMALLINT,
    created_at TIMESTAMP DEFAULT NOW()
);

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

CREATE TABLE IF NOT EXISTS matching_results (
    id SERIAL PRIMARY KEY,
    image_a_id INTEGER REFERENCES images(id) ON DELETE CASCADE,
    image_b_id INTEGER REFERENCES images(id) ON DELETE CASCADE,
    score FLOAT,
    method VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);
