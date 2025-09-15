# Therapist RAG System

Therapeutic guidance and case search system using PostgreSQL + OpenAI.

## Requirements

- Docker
- Python 3.8+
- OpenAI API Key

## Step by Step Installation

### Step 1: Configure OpenAI API Key

Edit the `.env` file and replace the API key:

```bash
OPENAI_API_KEY=your_actual_openai_api_key_here
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Start PostgreSQL Database

```bash
docker-compose up -d postgres
```

Wait for PostgreSQL to start (about 10 seconds):

```bash
sleep 10
```

### Step 4: Process the Data

```bash
python3 -m src.data_processor
```

### Step 5: Generate Embeddings

```bash
python3 -m src.embeddings
```

### Step 6: Start the Application

```bash
python3 main.py
```

### Step 7: Access the System

Open your browser and go to:

```
http://localhost:8001
```

## API Documentation

```
http://localhost:8001/docs
```

## Health Check

```
http://localhost:8001/api/v1/health
```

## Features

- Search similar therapy cases
- Generate AI-powered therapeutic guidance
- View system statistics

## Troubleshooting

### OpenAI API Key Error

Make sure you have set your real API key in the `.env` file:

```bash
OPENAI_API_KEY=sk-your-real-key-here
```

### PostgreSQL Connection Error

Restart the database:

```bash
docker-compose down
docker-compose up -d postgres
sleep 10
```


## Stop the System

To stop PostgreSQL:

```bash
docker-compose down
```
