# hyni_py
Hyni Python - Dynamic, schema-based context management for chat APIs

*Write once, run anywhere. Switch between OpenAI, Claude, DeepSeek, and more with zero code changes.*

---

## ✨ Why Hyni Python?
```python
# One line to rule them all - works with ANY LLM provider
from hyni_py import ChatApiBuilder, SchemaRegistry

# Create registry once and reuse
registry = SchemaRegistry.create().set_schema_directory("./schemas").build()

# Create a chat API for any provider
chat = (ChatApiBuilder()
        .schema("schemas/claude.json")
        .api_key(api_key)
        .config({"default_temperature": 0.3, "default_max_tokens": 500})
        .build())

# Set system message and send your query
chat.context.set_system_message("You are a helpful professor in quantum theory")
answer = chat.send_message("What is quantum mechanics?")
```
**No more provider-specific code.** No more JSON wrestling. Just pure, elegant AI conversations.

---

## 🚀 Features
- 🎯 Provider Agnostic - Switch between OpenAI, Claude, DeepSeek with one line
- 🧠 Smart Context Management - Automatic conversation history and memory
- 🎙️ Live Audio Transcription - Built-in Whisper integration (TODO)
- 💬 Streaming & Async - Real-time responses with full async support
- 🛑 Cancellation Control - Stop requests mid-flight
- 📦 Modern Python - Clean, expressive API design with type hints
- 🔧 Schema-Driven - JSON configs handle all provider differences


## 🎯 Quick Start

### The Simplest Way
```python
from hyni_py import GeneralContext, ChatApi

# Create a context and chat API
context = GeneralContext("schemas/claude.json")
chat = ChatApi(context)
chat.context.set_api_key("XYZ").set_system_message("You are a friendly software engineer")

# One-liner conversation
answer = chat.send_message("What is recursion?")
```

### Fluent Builder Pattern
```python
from hyni_py import ChatApiBuilder

# Use the builder for clear, readable configuration
chat = (ChatApiBuilder()
        .schema("schemas/openai.json")  # Required first!
        .api_key("your-api-key")        # Optional, any order
        .config({                       # Optional, any order
            "default_temperature": 0.8,
            "default_max_tokens": 1000
        })
        .build())

response = chat.send_message("What's the difference between a macchiato and a cortado?")
```

### Thread-Safe Context for Multi-Threaded Apps
```python
from hyni_py import SchemaRegistry, ContextFactory
import threading

# Create factory once at application startup
registry = SchemaRegistry.create().set_schema_directory("./schemas").build()
factory = ContextFactory(registry)

# Thread-local storage for contexts
local_contexts = threading.local()

def get_thread_local_context(provider_name):
    if not hasattr(local_contexts, provider_name):
        setattr(local_contexts, provider_name, factory.create_context(provider_name))
    return getattr(local_contexts, provider_name)

# In each worker thread
def process_query(query):
    # Get thread-local context - created once per thread
    context = get_thread_local_context("claude")
    context.set_api_key(get_api_key())
    
    chat = ChatApi(context)  
    response = chat.send_message(query)
    
    # Process response...
```

### Reuse the same chat for multiple or multi-turn questions
```python
response = chat.send_message("Write a Python class for a stack")
chat.context.add_assistant_message(response)
response = chat.send_message("Now add error handling")
chat.context.add_assistant_message(response)
response = chat.send_message("Show me how to test it")
chat.context.add_assistant_message(response)
response = chat.send_message("And now, explain me the time and space complexities")
```

### Advanced Context Management
```python
from hyni_py import GeneralContext, ChatApi

context = GeneralContext("schemas/claude.json")
context.set_system_message("You are an expert in German tax system")

# Build conversations naturally
context.add_user_message("Can you help me?")
context.add_assistant_message("I'd be happy to help! What's your question?")
context.add_user_message("What is the difference between child allowance and child benefit in tax matters?")

chat = ChatApi(context)
response = chat.send_message()  # Send with existing context
```

### Multimodal Conversations with Images
```python
from hyni_py import ChatApiBuilder
import base64

# Create a context with a provider that supports multimodal inputs
chat = (ChatApiBuilder()
        .schema("schemas/claude.json")  # Claude 3 supports images
        .api_key("your-api-key")
        .build())

# Method 1: Direct file path
chat.context.add_user_message(
    "What can you tell me about this image?", 
    "image/png", 
    "path/to/your/image.png"
)

# Method 2: Base64-encoded image data
with open("image.png", "rb") as f:
    base64_data = base64.b64encode(f.read()).decode('utf-8')
    
chat.context.add_user_message(
    "Describe this chart in detail and explain what it means.", 
    "image/jpeg", 
    base64_data
)

# Method 3: With streaming response for multimodal
def on_chunk(chunk):
    print(chunk, end="", flush=True)

chat.send_message_stream(
    "What text appears in this image?",
    "image/png",
    "screenshots/error_message.png",
    on_chunk=on_chunk
)

# Method 4: Multiple images in one conversation
context = chat.context
context.add_user_message("I'll show you two images. Compare them.")
context.add_user_message("Here's the first one:", "image/png", "product_v1.png")
context.add_assistant_message("I see the first image. It shows a product with...")
context.add_user_message("Here's the second one:", "image/png", "product_v2.png")

comparison = chat.send_message("What are the key differences?")
```

## 🔄 Sync vs Async - Your Choice
### Synchronous (Simple)
```python
from hyni_py import GeneralContext, ChatApi

context = GeneralContext("schemas/deepseek.json")
chat = ChatApi(context)

result = chat.send_message("Explain how to breed kois")
print(result)
```

### Asynchronous (Powerful)
```python
from hyni_py import GeneralContext, ChatApi
import time

context = GeneralContext("schemas/claude.json")
chat = ChatApi(context)

future = chat.send_message_async("Generate a story about AI")

# Do other work...
time.sleep(1)

# Get result when ready
story = future.result()
```

### Async/Await (Modern)
```python
from hyni_py import GeneralContext, ChatApi
import asyncio

async def main():
    context = GeneralContext("schemas/claude.json")
    chat = ChatApi(context)
    
    response = await chat.send_message_async_await("Generate a story about AI")
    print(response)
    
asyncio.run(main())
```

### Streaming (Real-time)
```python
from hyni_py import GeneralContext, ChatApi

context = GeneralContext("schemas/openai.json")
chat = ChatApi(context)

def on_chunk(chunk):
    print(chunk, end="", flush=True)  # Print as it arrives
    return True  # Continue streaming

def on_complete(response):
    print("\n--- Complete! ---")

chat.send_message_stream(
    "Write a long technical article",
    on_chunk=on_chunk,
    on_complete=on_complete
)
```

### Cancellable Operations
```python
from hyni_py import GeneralContext, ChatApi
import threading
import time

context = GeneralContext("schemas/openai.json")
chat = ChatApi(context)

should_cancel = False

def cancel_check():
    return should_cancel

future = chat.send_message_async("This might take a while...")

# Cancel after 5 seconds
time.sleep(5)
should_cancel = True
```

## 🎨 Multiple Ways to Build Conversations
### 1. Direct Message Sending
```python
from hyni_py import GeneralContext, ChatApi

context = GeneralContext("schemas/claude.json")
context.set_system_message("You are a senior Python developer")

chat = ChatApi(context)
response = chat.send_message("Implement quicksort in Python")
```

### 2. Context-First Approach (Recommended)
```python
from hyni_py import GeneralContext, ChatApi

context = GeneralContext("schemas/openai.json")
context.set_system_message("You are a creative writer")
context.add_user_message("Write a haiku about programming")

chat = ChatApi(context)
poem = chat.send_message()  # Uses existing context
```

### 3. Conversational Building
```python
from hyni_py import GeneralContext, ChatApi

context = GeneralContext("schemas/claude.json")
chat = ChatApi(context)

# Build conversation step by step
chat.context.add_user_message("Hello!")
chat.context.add_assistant_message("Hi! How can I help you?")
chat.context.add_user_message("Tell me about the meaning of life")

response = chat.send_message()
```

### 4. Parameter Configuration
```python
from hyni_py import GeneralContext, ChatApi

context = GeneralContext("schemas/claude.json")
context.set_parameter("temperature", 0.3)
context.set_parameter("max_tokens", 500)
context.set_parameter("top_p", 0.9)

chat = ChatApi(context)
response = chat.send_message("Explain machine learning")
```

## 🔧 Provider Configuration
### Dynamic Provider Switching
```python
from hyni_py import GeneralContext, ChatApi

# Create contexts for different providers
openai_context = GeneralContext("schemas/openai.json")
claude_context = GeneralContext("schemas/claude.json")
deepseek_context = GeneralContext("schemas/deepseek.json")

# Same question, different providers
question = "Explain machine learning"

openai_chat = ChatApi(openai_context)
claude_chat = ChatApi(claude_context)
deepseek_chat = ChatApi(deepseek_context)

openai_view = openai_chat.send_message(question)
claude_view = claude_chat.send_message(question)
deepseek_view = deepseek_chat.send_message(question)
```

### Using the Registry for Provider Management
```python
from hyni_py import SchemaRegistry, ContextFactory, ChatApi

# Create registry once
registry = SchemaRegistry.create().set_schema_directory("./schemas").build()
factory = ContextFactory(registry)

# Get available providers
providers = registry.get_available_providers()
print("Available providers:")
for provider in providers:
    print(f"- {provider}")

# Create context for any available provider
context = factory.create_context("claude")
chat = ChatApi(context)

response = chat.send_message("How do I design a scalable API?")
```

## 📐 Architecture
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Your Code     │──▶│  Hyni Context    │──▶│  LLM Provider   │
│                 │    │  Management      │    │  (Any/All)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌────────▼────────┐              │
         │              │ Schema Registry │              │
         │              │ (JSON Config)   │              │
         │              └─────────────────┘              │
         │                       │                       │
         │              ┌────────▼────────┐              │
         │              │ Context Factory │              │
         │              └─────────────────┘              │
         │                                               │
         ▼                                               ▼
┌─────────────────┐                           ┌─────────────────┐
│ Thread-Local    │                           │   Streaming     │
│ Contexts        │                           │   Response      │
└─────────────────┘                           └─────────────────┘
```

## Core Components
- GeneralContext - Smart conversation management with automatic history
- ChatApi - Provider-agnostic HTTP client with streaming support
- HttpClient - Abstraction over HTTP libraries (requests/httpx)
- SchemaEngine - JSON-driven provider configuration
- ChatApiBuilder - Fluent builder for clean configuration

## 🔄 Advanced Features
### Thread-Local Context with Provider Helper
```python
from hyni_py import SchemaRegistry, ContextFactory, ChatApi
import threading

# Create once at application startup
registry = SchemaRegistry.create().set_schema_directory("./schemas").build()
factory = ContextFactory(registry)

# Thread-local storage
thread_local = threading.local()

class ProviderContext:
    def __init__(self, factory, provider_name):
        self.factory = factory
        self.provider_name = provider_name
        
    def get(self):
        if not hasattr(thread_local, self.provider_name):
            setattr(thread_local, self.provider_name, 
                   self.factory.create_context(self.provider_name))
        return getattr(thread_local, self.provider_name)

# Create provider context helpers
openai_ctx = ProviderContext(factory, "openai")
claude_ctx = ProviderContext(factory, "claude")

# In any thread:
def process_request(query):
    # Get thread-local context for claude
    context = claude_ctx.get()
    context.set_api_key(get_api_key())
    
    chat = ChatApi(context)
    response = chat.send_message(query)
    
    # Process response...
```

### Conversation State Management
```python
from hyni_py import GeneralContext, ChatApi
import json

context = GeneralContext("schemas/claude.json")
chat = ChatApi(context)

# Build up conversation history
chat.send_message("What are the SOLID principles?")
chat.send_message("Can you give me an example of the Single Responsibility Principle?")
chat.send_message("How does this apply to Python classes?")

# Export conversation state for persistence
state = chat.context.export_state()
with open("conversation.json", "w") as f:
    json.dump(state, f)

# Later, import the state to continue
new_context = GeneralContext("schemas/claude.json")
with open("conversation.json", "r") as f:
    saved_state = json.load(f)
new_context.import_state(saved_state)

new_chat = ChatApi(new_context)
response = new_chat.send_message("Can we apply this to microservices?")
```

### Stream Processing with Completion Callbacks
```python
from hyni_py import GeneralContext, ChatApi

context = GeneralContext("schemas/claude.json")
chat = ChatApi(context)

accumulated_response = []

def on_chunk(chunk):
    print(chunk, end="", flush=True)
    accumulated_response.append(chunk)
    return True  # Continue streaming

def on_complete(response):
    print("\n\n--- Stream Complete ---")
    print(f"Total length: {sum(len(c) for c in accumulated_response)} characters")
    
    # Save to file, log, or process final response
    with open("raii_explanation.txt", "w") as f:
        f.write("".join(accumulated_response))

chat.send_message_stream(
    "Write a detailed explanation of dependency injection",
    on_chunk=on_chunk,
    on_complete=on_complete
)
```

### Error Handling and Cancellation
```python
from hyni_py import GeneralContext, ChatApi, ChatApiError
import time
import threading

context = GeneralContext("schemas/claude.json")
chat = ChatApi(context)

user_cancelled = False

try:
    future = chat.send_message_async("Generate a very long story...")
    
    # Simulate user cancellation after 3 seconds
    def cancel_after_delay():
        global user_cancelled
        time.sleep(3)
        user_cancelled = True
    
    threading.Thread(target=cancel_after_delay).start()
    
    result = future.result(timeout=10)  # Wait up to 10 seconds
    
except Exception as e:
    print(f"Request failed or was cancelled: {str(e)}")
```

### 🛠️ Error Handling
```python
from hyni_py import GeneralContext, ChatApi, ChatApiError, StreamingNotSupportedError

context = GeneralContext("schemas/claude.json")
chat = ChatApi(context)

try:
    response = chat.send_message("Hello world")
    print(response)
    
except StreamingNotSupportedError:
    print("This provider doesn't support streaming")
except ChatApiError as e:
    print(f"API error: {str(e)}")
except Exception as e:
    print(f"General error: {str(e)}")

# With streaming and cancellation
should_stop = False

def on_chunk(chunk):
    print(chunk, end="", flush=True)
    return not should_stop  # Continue if not stopped

def on_complete(response):
    print("\nGeneration complete!")

def cancel_check():
    return should_stop

chat.send_message_stream(
    "Long generation task...",
    on_chunk=on_chunk,
    on_complete=on_complete,
    cancel_check=cancel_check
)
```

## 📋 Installation
```bash
# From PyPI
pip install hyni-py

# Development mode
git clone https://github.com/your-org/hyni_py.git
cd hyni_py
pip install -e .

# With extras
pip install hyni-py[httpx]  # Use httpx instead of requests
pip install hyni-py[all]    # Install all optional dependencies
```

## 🤝 Contributing
We love contributions! Whether it's new LLM providers, features, or bug fixes.

```bash
git clone https://github.com/your-org/hyni_py.git
cd hyni_py
pip install -e ".[dev]"
pytest
```

## 📄 License
MIT License - Use freely in commercial and open-source projects.

## Ready to supercharge your AI conversations?
```python
from hyni_py import ChatApiBuilder

chat = (ChatApiBuilder()
        .schema("schemas/claude.json")
        .api_key("your-api-key")
        .build())

response = chat.send_message("How do I get started with Hyni Python?")
# "Just create a context with a schema, build a ChatApi, and start chatting!"
```
