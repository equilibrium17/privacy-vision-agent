import os

# The suite must never reach a real provider. Pin the default before
# server.main runs load_dotenv(), which does not override variables that are
# already set; provider tests set AI_PROVIDER explicitly.
os.environ["AI_PROVIDER"] = "groq"
