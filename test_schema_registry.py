"""
Unit tests for SchemaRegistry, ContextFactory, and ProviderContext classes.
"""

import pytest
import json
import threading
import time
from pathlib import Path
from typing import Dict, Any

from general_context import GeneralContext, ContextConfig, SchemaException
from schema_registry import (
    SchemaRegistry, ContextFactory, ProviderContext, CacheStats,
    create_registry_from_directory, create_factory_with_defaults
)


class TestSchemaRegistry:
    """Tests for SchemaRegistry class."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Set up test environment before each test."""
        # Create test directory structure
        Path("test_schemas").mkdir(exist_ok=True)
        Path("custom_schemas").mkdir(exist_ok=True)

        # Create dummy schema files
        self.create_dummy_schema_file("test_schemas/provider1.json")
        self.create_dummy_schema_file("test_schemas/provider2.json")
        self.create_dummy_schema_file("custom_schemas/provider3.json")

        yield  # Run the test

        # Clean up test directories and files
        import shutil
        shutil.rmtree("test_schemas", ignore_errors=True)
        shutil.rmtree("custom_schemas", ignore_errors=True)

    def create_dummy_schema_file(self, path: str):
        """Create a dummy schema file for testing."""
        schema = {
            "provider": {"name": "test", "display_name": "Test AI"},
            "api": {"endpoint": "https://test.com/api"},
            "request_template": {},
            "message_format": {"structure": {}, "content_types": {}},
            "response_format": {
                "success": {"text_path": ["choices", 0, "message", "content"]}
            }
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(schema, f)

    def test_builder_pattern(self):
        """Test builder pattern for SchemaRegistry."""
        registry = (SchemaRegistry.builder()
                    .set_schema_directory("test_schemas")
                    .register_schema("custom_provider", "custom_schemas/provider3.json")
                    .build())

        assert registry is not None
        assert registry.is_provider_available("provider1")
        assert registry.is_provider_available("provider2")
        assert registry.is_provider_available("custom_provider")

    def test_resolve_schema_path(self):
        """Test path resolution for schemas."""
        registry = (SchemaRegistry.builder()
                    .set_schema_directory("test_schemas")
                    .register_schema("custom_provider", "custom_schemas/provider3.json")
                    .build())

        path1 = registry.resolve_schema_path("provider1")
        path2 = registry.resolve_schema_path("custom_provider")

        assert path1.is_absolute()
        assert path2.is_absolute()
        assert "test_schemas/provider1.json" in str(path1)
        assert "custom_schemas/provider3.json" in str(path2)

    def test_empty_provider_name(self):
        """Test handling of empty provider name."""
        registry = SchemaRegistry.builder().build()

        with pytest.raises(ValueError):
            registry.resolve_schema_path("")

        with pytest.raises(ValueError):
            registry.is_provider_available("")

    def test_get_available_providers(self):
        """Test getting list of available providers."""
        registry = (SchemaRegistry.builder()
                    .set_schema_directory("test_schemas")
                    .register_schema("custom_provider", "custom_schemas/provider3.json")
                    .build())

        providers = registry.get_available_providers()

        assert len(providers) == 3
        assert "provider1" in providers
        assert "provider2" in providers
        assert "custom_provider" in providers

    def test_is_provider_available(self):
        """Test provider availability check."""
        registry = SchemaRegistry.builder().set_schema_directory("test_schemas").build()

        assert registry.is_provider_available("provider1")
        assert registry.is_provider_available("provider2")
        assert not registry.is_provider_available("nonexistent_provider")

    def test_registry_immutability(self):
        """Test immutability of registry after construction."""
        registry = SchemaRegistry.builder().set_schema_directory("test_schemas").build()

        with pytest.raises(AttributeError):
            registry._schema_directory = Path("new_path")

        with pytest.raises(AttributeError):
            registry._provider_paths = {}


class TestContextFactory:
    """Tests for ContextFactory class."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Set up test environment before each test."""
        # Create test directory structure
        Path("test_schemas").mkdir(exist_ok=True)
        Path("custom_schemas").mkdir(exist_ok=True)

        # Create dummy schema files
        self.create_dummy_schema_file("test_schemas/provider1.json")
        self.create_dummy_schema_file("test_schemas/provider2.json")
        self.create_dummy_schema_file("custom_schemas/provider3.json")

        # Create registry and factory
        self.registry = (SchemaRegistry.builder()
                         .set_schema_directory("test_schemas")
                         .register_schema("custom_provider", "custom_schemas/provider3.json")
                         .build())
        self.factory = ContextFactory(self.registry)

        yield  # Run the test

        # Clean up test directories and files
        import shutil
        shutil.rmtree("test_schemas", ignore_errors=True)
        shutil.rmtree("custom_schemas", ignore_errors=True)

    def create_dummy_schema_file(self, path: str):
        """Create a dummy schema file for testing."""
        schema = {
            "provider": {"name": "test", "display_name": "Test AI"},
            "api": {"endpoint": "https://test.com/api"},
            "request_template": {},
            "message_format": {"structure": {}, "content_types": {}},
            "response_format": {
                "success": {"text_path": ["choices", 0, "message", "content"]}
            }
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(schema, f)

    def test_create_context(self):
        """Test context creation."""
        # Should succeed for existing providers
        context1 = self.factory.create_context("provider1")
        assert context1 is not None
        assert context1.get_provider_name() == "test"

        # Should throw for non-existent providers
        with pytest.raises(SchemaException):
            self.factory.create_context("nonexistent_provider")

    def test_create_context_with_config(self):
        """Test context creation with configuration."""
        config = ContextConfig(
            default_max_tokens=100,
            default_temperature=0.7
        )

        context = self.factory.create_context("provider1", config)
        assert context is not None

        # Build request to check if config was applied
        context.add_user_message("Test")
        request = context.build_request()
        assert request["max_tokens"] == 100
        assert request["temperature"] == 0.7

    def test_schema_caching(self):
        """Test schema caching behavior."""
        # First creation should cache the schema
        context1 = self.factory.create_context("provider1")
        stats1 = self.factory.get_cache_stats()
        assert stats1.cache_size == 1
        assert stats1.hit_count == 0
        assert stats1.miss_count == 1

        # Second creation should use cached schema
        context2 = self.factory.create_context("provider1")
        stats2 = self.factory.get_cache_stats()
        assert stats2.cache_size == 1
        assert stats2.hit_count == 1
        assert stats2.miss_count == 1

        # Different provider should miss cache
        context3 = self.factory.create_context("provider2")
        stats3 = self.factory.get_cache_stats()
        assert stats3.cache_size == 2
        assert stats3.hit_count == 1
        assert stats3.miss_count == 2

    def test_clear_cache(self):
        """Test cache clearing functionality."""
        self.factory.create_context("provider1")
        self.factory.create_context("provider2")

        stats1 = self.factory.get_cache_stats()
        assert stats1.cache_size == 2

        self.factory.clear_cache()

        stats2 = self.factory.get_cache_stats()
        assert stats2.cache_size == 0
        assert stats2.hit_count == 0
        assert stats2.miss_count == 0

    def test_thread_local_context(self):
        """Test thread-local context behavior."""
        # Get thread-local context
        context1 = self.factory.get_thread_local_context("provider1")
        context2 = self.factory.get_thread_local_context("provider1")

        # Should be the same instance within the same thread
        assert context1 is context2

        # Test in another thread using an event to synchronize
        result = {}
        event = threading.Event()

        def thread_func():
            context_thread = self.factory.get_thread_local_context("provider1")
            result['thread_context'] = context_thread
            event.set()

        thread = threading.Thread(target=thread_func)
        thread.start()
        event.wait()
        thread.join()

        # Address should be different from main thread's context
        assert result['thread_context'] is not context1

    def test_multi_threaded_access(self):
        """Test multi-threaded access to factory."""
        NUM_THREADS = 10
        success_count = 0
        lock = threading.Lock()
        event = threading.Event()
        threads = []

        def worker():
            nonlocal success_count
            try:
                context = self.factory.create_context("provider1")
                tl_context = self.factory.get_thread_local_context("provider2")
                with lock:
                    success_count += 1
            except Exception as e:
                pytest.fail(f"Exception in thread: {e}")

        for _ in range(NUM_THREADS):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert success_count == NUM_THREADS

        stats = self.factory.get_cache_stats()
        assert stats.cache_size == 2  # Only two unique providers used

    def test_null_registry(self):
        """Test null registry handling."""
        with pytest.raises(ValueError):
            ContextFactory(None)

    def test_invalid_schema_file(self):
        """Test invalid schema file handling."""
        # Create invalid JSON file
        with open("test_schemas/invalid.json", 'w') as f:
            f.write("{ invalid json")

        with pytest.raises(SchemaException):
            self.factory.create_context("invalid")

    def test_cache_stats_properties(self):
        """Test cache stats properties."""
        # Create some contexts to populate cache
        self.factory.create_context("provider1")
        self.factory.create_context("provider1")  # Hit
        self.factory.create_context("provider2")  # Miss

        stats = self.factory.get_cache_stats()
        assert stats.hit_count == 1
        assert stats.miss_count == 2
        assert stats.hit_rate == 1/3


class TestProviderContext:
    """Tests for ProviderContext class."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Set up test environment before each test."""
        # Create test directory structure
        Path("test_schemas").mkdir(exist_ok=True)

        # Create dummy schema file
        self.create_dummy_schema_file("test_schemas/provider1.json")

        # Create registry and factory
        self.registry = SchemaRegistry.builder().set_schema_directory("test_schemas").build()
        self.factory = ContextFactory(self.registry)
        self.provider_context = ProviderContext(self.factory, "provider1")

        yield  # Run the test

        # Clean up test directories and files
        import shutil
        shutil.rmtree("test_schemas", ignore_errors=True)

    def create_dummy_schema_file(self, path: str):
        """Create a dummy schema file for testing."""
        schema = {
            "provider": {"name": "test", "display_name": "Test AI"},
            "api": {"endpoint": "https://test.com/api"},
            "request_template": {},
            "message_format": {"structure": {}, "content_types": {}},
            "response_format": {
                "success": {"text_path": ["choices", 0, "message", "content"]}
            }
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(schema, f)

    def test_provider_context_helper(self):
        """Test provider context helper functionality."""
        # First access creates the context
        context1 = self.provider_context.get()
        assert context1.get_provider_name() == "test"

        # Second access reuses it
        context2 = self.provider_context.get()
        assert context1 is context2

    def test_reset(self):
        """Test context reset functionality."""
        context = self.provider_context.get()
        context.add_user_message("Hello")
        assert len(context.get_messages()) > 0

        self.provider_context.reset()
        assert len(context.get_messages()) == 0

    def test_provider_name_property(self):
        """Test provider name property."""
        assert self.provider_context.provider_name == "provider1"

    def test_thread_local_behavior(self):
        """Test thread-local behavior of provider context."""
        context_main = self.provider_context.get()
        context_main.add_user_message("Main thread message")

        result = {}
        event = threading.Event()

        def thread_func():
            thread_context = self.provider_context.get()
            result['thread_context'] = thread_context
            result['thread_messages'] = len(thread_context.get_messages())
            event.set()

        thread = threading.Thread(target=thread_func)
        thread.start()
        event.wait()
        thread.join()

        # Should be different instances
        assert result['thread_context'] is not context_main
        # Thread context should start empty
        assert result['thread_messages'] == 0
        # Main thread context should keep its messages
        assert len(context_main.get_messages()) == 1


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Set up test environment before each test."""
        # Create test directory structure
        Path("test_schemas").mkdir(exist_ok=True)
        self.create_dummy_schema_file("test_schemas/provider1.json")

        yield  # Run the test

        # Clean up test directories and files
        import shutil
        shutil.rmtree("test_schemas", ignore_errors=True)

    def create_dummy_schema_file(self, path: str):
        """Create a dummy schema file for testing."""
        schema = {
            "provider": {"name": "test", "display_name": "Test AI"},
            "api": {"endpoint": "https://test.com/api"},
            "request_template": {},
            "message_format": {"structure": {}, "content_types": {}},
            "response_format": {
                "success": {"text_path": ["choices", 0, "message", "content"]}
            }
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(schema, f)

    def test_create_registry_from_directory(self):
        """Test creating registry from directory."""
        registry = create_registry_from_directory("test_schemas")
        assert registry is not None
        assert registry.is_provider_available("provider1")

    def test_create_factory_with_defaults(self):
        """Test creating factory with default settings."""
        factory = create_factory_with_defaults("test_schemas", cache_size=64)
        assert factory is not None

        # Should be able to create context
        context = factory.create_context("provider1")
        assert context is not None
        assert context.get_provider_name() == "test"

        # Cache size should be limited
        for i in range(100):
            # Create a temporary file for this test
            path = f"test_schemas/temp{i}.json"
            self.create_dummy_schema_file(path)
            try:
                factory.create_context(f"temp{i}")
            except:
                pass  # Ignore errors, we just want to fill the cache

        stats = factory.get_cache_stats()
        assert stats.cache_size <= 64  # LRU cache should limit size


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Set up test environment before each test."""
        Path("test_schemas").mkdir(exist_ok=True)
        self.create_dummy_schema_file("test_schemas/provider1.json")

        yield

        import shutil
        shutil.rmtree("test_schemas", ignore_errors=True)

    def create_dummy_schema_file(self, path: str):
        """Create a dummy schema file for testing."""
        schema = {
            "provider": {"name": "test", "display_name": "Test AI"},
            "api": {"endpoint": "https://test.com/api"},
            "request_template": {},
            "message_format": {"structure": {}, "content_types": {}},
            "response_format": {
                "success": {"text_path": ["choices", 0, "message", "content"]}
            }
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(schema, f)

    def test_nonexistent_directory(self):
        """Test handling of nonexistent directory."""
        registry = SchemaRegistry.builder().set_schema_directory("nonexistent_dir").build()
        assert registry.get_available_providers() == []

    def test_empty_schema_directory(self):
        """Test handling of empty schema directory."""
        import shutil
        shutil.rmtree("test_schemas", ignore_errors=True)
        Path("test_schemas").mkdir(exist_ok=True)

        registry = SchemaRegistry.builder().set_schema_directory("test_schemas").build()
        assert registry.get_available_providers() == []

    def test_invalid_json_schema(self):
        """Test handling of invalid JSON schema."""
        with open("test_schemas/invalid.json", 'w') as f:
            f.write("{ this is not valid json }")

        registry = SchemaRegistry.builder().set_schema_directory("test_schemas").build()
        factory = ContextFactory(registry)

        with pytest.raises(SchemaException):
            factory.create_context("invalid")

    def test_cache_limit(self):
        """Test cache size limit enforcement."""
        factory = ContextFactory(
            SchemaRegistry.builder().set_schema_directory("test_schemas").build(),
            cache_size=2
        )

        # Create 3 different contexts to exceed cache size
        for i in range(3):
            path = f"test_schemas/provider{i}.json"
            self.create_dummy_schema_file(path)
            factory.create_context(f"provider{i}")

        stats = factory.get_cache_stats()
        assert stats.cache_size <= 2

    def test_registry_builder_validation(self):
        """Test validation in registry builder."""
        builder = SchemaRegistry.builder()

        with pytest.raises(ValueError):
            builder.register_schema("", "path.json")

    def test_concurrent_cache_access(self):
        """Test concurrent access to the cache."""
        factory = ContextFactory(
            SchemaRegistry.builder().set_schema_directory("test_schemas").build()
        )

        # Create many threads accessing the cache simultaneously
        NUM_THREADS = 20
        threads = []

        for i in range(NUM_THREADS):
            t = threading.Thread(
                target=lambda: factory.create_context("provider1")
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Cache should be consistent - we can't guarantee exact hit/miss counts due to race conditions
        stats = factory.get_cache_stats()
        assert stats.cache_size == 1  # We should have exactly one cached schema
        assert stats.hit_count + stats.miss_count == NUM_THREADS  # Total operations should match thread count
        assert stats.miss_count >= 1  # At least one miss (first access)
        assert stats.miss_count <= 3  # But not too many misses (allowing for some race conditions)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
