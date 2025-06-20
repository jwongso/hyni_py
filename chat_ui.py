"""
PyHyni - A PyQt6 GUI for interacting with various Language Models
Supports multiple providers through schema-based configuration
"""

import sys
import os
import json
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QMenuBar, QMenu, QTextBrowser,
    QSplitter, QMessageBox, QInputDialog, QLabel, QComboBox,
    QFileDialog, QProgressDialog, QCheckBox, QDialog, QDialogButtonBox,
    QPlainTextEdit
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSettings
from PyQt6.QtGui import QAction, QFont, QTextCursor, QKeySequence, QTextDocument

# Import the chat API modules
from chat_api import ChatApi, ChatApiBuilder, StreamingNotSupportedError
from general_context import GeneralContext, ContextConfig

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pyhyni.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class SystemMessageDialog(QDialog):
    """Dialog to set system message"""
    def __init__(self, current_message: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set System Message")
        self.setModal(True)
        self.resize(600, 400)

        layout = QVBoxLayout()

        # Instructions
        instructions = QLabel("Enter system message (instructions for the AI):")
        layout.addWidget(instructions)

        # Text editor
        self.text_edit = QPlainTextEdit()
        self.text_edit.setPlainText(current_message)
        self.text_edit.setFont(QFont("Arial", 10))
        layout.addWidget(self.text_edit)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def get_system_message(self) -> str:
        return self.text_edit.toPlainText().strip()


class DebugDialog(QDialog):
    """Dialog to show debug information"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Debug Information")
        self.setModal(True)
        self.resize(600, 400)

        layout = QVBoxLayout()

        self.text_browser = QTextBrowser()
        self.text_browser.setFont(QFont("Courier", 9))
        layout.addWidget(self.text_browser)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def set_debug_info(self, info: str):
        self.text_browser.setPlainText(info)


def get_api_key_for_provider(provider: str) -> str:
    """
    Get API key for a specific provider from environment variables.

    Args:
        provider: Provider name (e.g., 'claude', 'openai', 'deepseek', 'mistral')

    Returns:
        API key string or empty string if not found
    """
    logger.debug(f"Looking for API key for provider: {provider}")

    # Updated mapping with new environment variable names
    env_mapping = {
        'claude': 'CL_API_KEY',
        'openai': 'OA_API_KEY',
        'deepseek': 'DS_API_KEY',
        'mistral': 'MS_API_KEY'
    }

    env_var = env_mapping.get(provider.lower(), f"{provider.upper()}_API_KEY")
    logger.debug(f"Checking environment variable: {env_var}")

    api_key = os.environ.get(env_var, '')

    if api_key:
        logger.info(f"Found API key for {provider} in environment variable {env_var}")
        return api_key

    # Try loading from .hynirc if not in environment
    rc_path = Path.home() / '.hynirc'
    logger.debug(f"Checking .hynirc file at: {rc_path}")

    if rc_path.exists():
        try:
            with open(rc_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(f"export {env_var}="):
                        api_key = line.split('=', 1)[1].strip().strip('"').strip("'")
                        logger.info(f"Found API key for {provider} in .hynirc file")
                        return api_key
                    elif line.startswith(f"{env_var}="):
                        api_key = line.split('=', 1)[1].strip().strip('"').strip("'")
                        logger.info(f"Found API key for {provider} in .hynirc file")
                        return api_key
        except Exception as e:
            logger.error(f"Error reading .hynirc file: {e}")

    logger.warning(f"No API key found for provider: {provider}")
    return api_key


def get_env_var_name(provider: str) -> str:
    """Get the environment variable name for a provider"""
    env_mapping = {
        'claude': 'CL_API_KEY',
        'openai': 'OA_API_KEY',
        'deepseek': 'DS_API_KEY',
        'mistral': 'MS_API_KEY'
    }
    return env_mapping.get(provider.lower(), f"{provider.upper()}_API_KEY")


class ProviderInfo:
    """Container for provider information loaded from schema"""
    def __init__(self, schema_path: str, schema_data: Dict[str, Any]):
        self.schema_path = schema_path
        self.schema_data = schema_data

        # Extract provider info
        provider = schema_data.get("provider", {})
        self.name = provider.get("name", Path(schema_path).stem)
        self.display_name = provider.get("display_name", self.name)
        self.version = provider.get("version", "1.0")

        logger.info(f"Loaded provider: {self.display_name} (name: {self.name}, version: {self.version})")

        # Extract API endpoint
        api = schema_data.get("api", {})
        self.endpoint = api.get("endpoint", "")
        logger.debug(f"Provider {self.name} endpoint: {self.endpoint}")

        # Extract models
        models = schema_data.get("models", {})
        self.available_models = models.get("available", [])
        self.default_model = models.get("default", self.available_models[0] if self.available_models else None)

        logger.debug(f"Provider {self.name} models: {self.available_models}, default: {self.default_model}")

        # Extract features
        features = schema_data.get("features", {})
        self.supports_streaming = features.get("streaming", False)
        self.supports_vision = features.get("vision", False)
        self.supports_system_messages = features.get("system_messages", False)

        logger.debug(f"Provider {self.name} features - streaming: {self.supports_streaming}, "
                    f"vision: {self.supports_vision}, system_messages: {self.supports_system_messages}")

        # Extract authentication info
        auth = schema_data.get("authentication", {})
        self.auth_type = auth.get("type", "header")
        self.key_name = auth.get("key_name", "Authorization")
        self.key_prefix = auth.get("key_prefix", "")


class SchemaLoader(QThread):
    """Thread for loading schemas from a directory"""
    provider_loaded = pyqtSignal(ProviderInfo)
    error_occurred = pyqtSignal(str)
    finished_loading = pyqtSignal()

    def __init__(self, schema_dir: str):
        super().__init__()
        self.schema_dir = schema_dir
        logger.info(f"Initializing schema loader for directory: {schema_dir}")

    def run(self):
        try:
            schema_path = Path(self.schema_dir)
            if not schema_path.exists() or not schema_path.is_dir():
                error_msg = f"Invalid schema directory: {self.schema_dir}"
                logger.error(error_msg)
                self.error_occurred.emit(error_msg)
                return

            # Load all JSON files in the directory
            json_files = list(schema_path.glob("*.json"))
            logger.info(f"Found {len(json_files)} JSON files in {self.schema_dir}")

            for json_file in json_files:
                try:
                    logger.debug(f"Loading schema file: {json_file}")
                    with open(json_file, 'r', encoding='utf-8') as f:
                        schema_data = json.load(f)

                    # Validate it's a valid schema
                    if "provider" in schema_data and "api" in schema_data:
                        provider_info = ProviderInfo(str(json_file), schema_data)
                        self.provider_loaded.emit(provider_info)
                    else:
                        logger.warning(f"Invalid schema file (missing provider or api): {json_file}")
                except Exception as e:
                    error_msg = f"Error loading {json_file.name}: {str(e)}"
                    logger.error(error_msg)
                    self.error_occurred.emit(error_msg)

            logger.info("Finished loading all schemas")
            self.finished_loading.emit()

        except Exception as e:
            error_msg = f"Error scanning directory: {str(e)}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)


class StreamingWorker(QThread):
    """Worker thread for handling streaming responses"""
    chunk_received = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    finished_streaming = pyqtSignal()

    def __init__(self, api: ChatApi, message: str, keep_history: bool = False):
        super().__init__()
        self.api = api
        self.message = message
        self.keep_history = keep_history
        self._is_cancelled = False
        self._accumulated_response = ""
        logger.info(f"Created streaming worker for message (keep_history={keep_history})")

    def run(self):
        try:
            logger.info("Starting streaming request")

            def on_chunk(chunk: str):
                if not self._is_cancelled:
                    logger.debug(f"Received chunk: {chunk[:50]}...")
                    self._accumulated_response += chunk
                    self.chunk_received.emit(chunk)

            def on_complete(response):
                if not self._is_cancelled:
                    logger.info("Streaming completed")
                    # Add assistant message to history if multi-turn is enabled
                    if self.keep_history and self._accumulated_response:
                        self.api.context.add_assistant_message(self._accumulated_response)
                        logger.debug("Added assistant response to conversation history")
                    self.finished_streaming.emit()

            def cancel_check():
                return self._is_cancelled

            self.api.send_message_stream(
                self.message,
                on_chunk=on_chunk,
                on_complete=on_complete,
                cancel_check=cancel_check
            )
        except StreamingNotSupportedError:
            error_msg = "Streaming is not supported by this provider"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Streaming error: {error_msg}")
            self.error_occurred.emit(error_msg)

    def cancel(self):
        logger.info("Cancelling streaming request")
        self._is_cancelled = True


class NonStreamingWorker(QThread):
    """Worker thread for handling non-streaming responses"""
    response_received = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, api: ChatApi, message: str, keep_history: bool = False):
        super().__init__()
        self.api = api
        self.message = message
        self.keep_history = keep_history
        self._is_cancelled = False
        logger.info(f"Created non-streaming worker for message (keep_history={keep_history})")

    def run(self):
        try:
            logger.info("Starting non-streaming request")

            def cancel_check():
                return self._is_cancelled

            response = self.api.send_message(self.message, cancel_check=cancel_check)
            if not self._is_cancelled:
                logger.info(f"Received response: {response[:100]}...")
                # Add assistant message to history if multi-turn is enabled
                if self.keep_history:
                    self.api.context.add_assistant_message(response)
                    logger.debug("Added assistant response to conversation history")
                self.response_received.emit(response)
        except Exception as e:
            if not self._is_cancelled:
                error_msg = str(e)
                logger.error(f"Non-streaming error: {error_msg}")
                self.error_occurred.emit(error_msg)

    def cancel(self):
        logger.info("Cancelling non-streaming request")
        self._is_cancelled = True


class ChatWidget(QWidget):
    """Main chat widget containing the conversation display and input area"""

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.current_response_position = None
        self.current_response_text = ""  # Store current response for markdown rendering
        self.use_markdown = True  # Enable markdown by default
        logger.info("Initialized chat widget with Markdown support")

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)

        # Conversation display - using QTextBrowser for better HTML/Markdown support
        self.conversation_display = QTextBrowser()
        self.conversation_display.setOpenExternalLinks(True)
        self.conversation_display.setFont(QFont("Arial", 10))

        # Input area with checkboxes
        input_layout = QHBoxLayout()

        self.input_text = QTextEdit()
        self.input_text.setMaximumHeight(100)
        self.input_text.setPlaceholderText("Type your message here...")
        self.input_text.setFont(QFont("Arial", 10))

        # Options layout
        options_layout = QVBoxLayout()

        # Streaming checkbox - OFF by default
        self.streaming_checkbox = QCheckBox("Stream")
        self.streaming_checkbox.setChecked(False)
        self.streaming_checkbox.setToolTip("Enable streaming responses (when supported)")

        # Multi-turn checkbox - ON by default for conversation continuity
        self.multiturn_checkbox = QCheckBox("Multi-turn")
        self.multiturn_checkbox.setChecked(True)
        self.multiturn_checkbox.setToolTip("Keep conversation history for context")

        # Markdown checkbox - ON by default
        self.markdown_checkbox = QCheckBox("Markdown")
        self.markdown_checkbox.setChecked(True)
        self.markdown_checkbox.setToolTip("Render responses as Markdown")
        self.markdown_checkbox.stateChanged.connect(self.on_markdown_toggle)

        options_layout.addWidget(self.streaming_checkbox)
        options_layout.addWidget(self.multiturn_checkbox)
        options_layout.addWidget(self.markdown_checkbox)

        self.send_button = QPushButton("Send")
        self.send_button.setMinimumHeight(40)
        self.send_button.setMinimumWidth(80)
        self.send_button.setEnabled(True)

        input_layout.addWidget(self.input_text, 1)
        input_layout.addLayout(options_layout)
        input_layout.addWidget(self.send_button)

        # Add widgets to layout
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.conversation_display)

        input_widget = QWidget()
        input_widget.setLayout(input_layout)
        splitter.addWidget(input_widget)

        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)
        self.setLayout(layout)

        # Connect Ctrl+Enter to send
        self.input_text.installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj == self.input_text and event.type() == event.Type.KeyPress:
            if event.key() == Qt.Key.Key_Return and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
                self.send_button.click()
                return True
        return super().eventFilter(obj, event)

    def on_markdown_toggle(self, state):
        """Handle markdown checkbox toggle"""
        self.use_markdown = state == Qt.CheckState.Checked.value
        logger.info(f"Markdown rendering: {'enabled' if self.use_markdown else 'disabled'}")

    def append_message(self, role: str, content: str, model_name: str = ""):
        """Append a message to the conversation display with improved formatting"""
        logger.debug(f"Appending {role} message: {content[:50]}...")

        cursor = self.conversation_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        # Add spacing between messages
        if cursor.position() > 0:
            cursor.insertHtml('<br>')

        # Add timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Create a container div for the message
        cursor.insertHtml('<div style="margin-bottom: 15px;">')

        # Format based on role
        if role == "user":
            cursor.insertHtml(f'<p style="color: #0066cc; margin: 5px 0; font-weight: bold;">You [{timestamp}]:</p>')
            # User messages are plain text
            escaped_content = content.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
            cursor.insertHtml(f'<div style="margin-left: 20px; padding: 10px; background-color: #f0f0f0; border-radius: 5px;">{escaped_content}</div>')

        elif role == "assistant":
            display_name = model_name if model_name else "Assistant"
            cursor.insertHtml(f'<p style="color: #009900; margin: 5px 0; font-weight: bold;">{display_name} [{timestamp}]:</p>')

            # Assistant messages can be markdown
            if self.use_markdown:
                cursor.insertHtml('<div style="margin-left: 20px; padding: 10px; background-color: #f8f8f8; border-radius: 5px;">')

                # Create a temporary document to render markdown
                temp_doc = QTextDocument()
                temp_doc.setMarkdown(content)
                cursor.insertHtml(temp_doc.toHtml())

                cursor.insertHtml('</div>')
            else:
                # Plain text fallback
                escaped_content = content.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
                cursor.insertHtml(f'<div style="margin-left: 20px; padding: 10px; background-color: #f8f8f8; border-radius: 5px;">{escaped_content}</div>')

        elif role == "error":
            cursor.insertHtml(f'<p style="color: #cc0000; margin: 5px 0; font-weight: bold;">Error [{timestamp}]:</p>')
            escaped_content = content.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
            cursor.insertHtml(f'<div style="margin-left: 20px; padding: 10px; background-color: #ffe0e0; border-radius: 5px;">{escaped_content}</div>')

        elif role == "system":
            cursor.insertHtml(f'<p style="color: #666666; margin: 5px 0; font-weight: bold;">System [{timestamp}]:</p>')
            escaped_content = content.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
            cursor.insertHtml(f'<div style="margin-left: 20px; padding: 10px; background-color: #e8e8e8; border-radius: 5px; color: #666666;">{escaped_content}</div>')

        cursor.insertHtml('</div>')

        # Scroll to bottom
        self.conversation_display.verticalScrollBar().setValue(
            self.conversation_display.verticalScrollBar().maximum()
        )

    def append_streaming_chunk(self, chunk: str, model_name: str = ""):
        """Append a chunk to the current streaming response"""
        cursor = self.conversation_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        # If this is the first chunk, add the assistant header
        if self.current_response_position is None:
            # Add spacing
            if cursor.position() > 0:
                cursor.insertHtml('<br>')

            timestamp = datetime.now().strftime("%H:%M:%S")
            display_name = model_name if model_name else "Assistant"

            cursor.insertHtml('<div style="margin-bottom: 15px;">')
            cursor.insertHtml(f'<p style="color: #009900; margin: 5px 0; font-weight: bold;">{display_name} [{timestamp}] (streaming):</p>')
            cursor.insertHtml('<div style="margin-left: 20px; padding: 10px; background-color: #f8f8f8; border-radius: 5px;" id="streaming-response">')

            self.current_response_position = cursor.position()
            self.current_response_text = ""

        # Accumulate the response text
        self.current_response_text += chunk

        # For streaming, we'll show plain text and render markdown when complete
        escaped_chunk = chunk.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        cursor.insertHtml(escaped_chunk)

        # Scroll to bottom
        self.conversation_display.verticalScrollBar().setValue(
            self.conversation_display.verticalScrollBar().maximum()
        )

    def finish_streaming_response(self):
        """Finish the current streaming response and optionally render as markdown"""
        if self.current_response_position is not None:
            cursor = self.conversation_display.textCursor()

            # If markdown is enabled, re-render the complete response
            if self.use_markdown and self.current_response_text:
                # Move to the start of the streaming response
                cursor.setPosition(self.current_response_position)
                cursor.movePosition(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor)

                # Replace with markdown-rendered version
                temp_doc = QTextDocument()
                temp_doc.setMarkdown(self.current_response_text)
                cursor.insertHtml(temp_doc.toHtml())

            # Close the divs
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertHtml('</div></div>')

            self.current_response_position = None
            self.current_response_text = ""
            logger.debug("Finished streaming response display")

    def clear_conversation(self):
        """Clear the conversation display"""
        logger.info("Clearing conversation")
        self.conversation_display.clear()
        self.current_response_position = None
        self.current_response_text = ""

    def get_input_text(self) -> str:
        """Get and clear the input text"""
        text = self.input_text.toPlainText().strip()
        self.input_text.clear()
        return text

    def is_streaming_enabled(self) -> bool:
        """Check if streaming is enabled"""
        return self.streaming_checkbox.isChecked()

    def is_multiturn_enabled(self) -> bool:
        """Check if multi-turn conversation is enabled"""
        return self.multiturn_checkbox.isChecked()

    def is_markdown_enabled(self) -> bool:
        """Check if markdown rendering is enabled"""
        return self.markdown_checkbox.isChecked()


class ApiKeyStatusWidget(QWidget):
    """Widget to display API key status"""

    def __init__(self):
        super().__init__()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        self.status_icon = QLabel()
        self.status_text = QLabel()

        layout.addWidget(self.status_icon)
        layout.addWidget(self.status_text)

        self.setLayout(layout)
        self.update_status(False, "")

    def update_status(self, has_key: bool, source: str = ""):
        """Update the API key status display"""
        if has_key:
            self.status_icon.setText("🔑")
            if source:
                self.status_text.setText(f"API Key: Available ({source})")
            else:
                self.status_text.setText("API Key: Available")
            self.status_text.setStyleSheet("color: green;")
        else:
            self.status_icon.setText("⚠️")
            self.status_text.setText("API Key: Not Set")
            self.status_text.setStyleSheet("color: orange;")


class PyHyniMainWindow(QMainWindow):
    """Main application window for PyHyni"""

    def __init__(self):
        super().__init__()
        logger.info("Initializing PyHyni")

        self.settings = QSettings("PyHyni", "GUI")

        # Provider management
        self.providers: Dict[str, ProviderInfo] = {}
        self.current_provider = None
        self.current_model = None
        self.schema_dir = self.settings.value("schema_dir", "schemas")

        logger.info(f"Schema directory: {self.schema_dir}")

        # API management
        self.api_keys = {}
        self.api_key_sources = {}
        self.chat_api = None

        # System message
        self.system_message = ""

        # Workers
        self.streaming_worker = None
        self.non_streaming_worker = None
        self.schema_loader = None

        self.init_ui()

        # Load schemas if directory exists
        if os.path.exists(self.schema_dir):
            self.load_schemas_from_directory(self.schema_dir)
        else:
            self.show_no_schemas_message()

    def init_ui(self):
        self.setWindowTitle("PyHyni - LLM Chat Interface")
        self.setGeometry(100, 100, 1000, 700)

        # Create central widget
        self.chat_widget = ChatWidget()
        self.setCentralWidget(self.chat_widget)

        # Create menu bar
        self.create_menu_bar()

        # Create status bar with enhanced layout
        self.status_label = QLabel("Ready")
        self.statusBar().addWidget(self.status_label)

        # API Key status widget
        self.api_key_status = ApiKeyStatusWidget()
        self.statusBar().addWidget(self.api_key_status)

        # Add stretch to push model selector to the right
        stretch_widget = QWidget()
        stretch_widget.setSizePolicy(
            stretch_widget.sizePolicy().horizontalPolicy(),
            stretch_widget.sizePolicy().verticalPolicy()
        )
        self.statusBar().addWidget(stretch_widget, 1)

        # Model selector in status bar
        self.model_combo = QComboBox()
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        self.statusBar().addPermanentWidget(QLabel("Model:"))
        self.statusBar().addPermanentWidget(self.model_combo)

        # Connect signals
        self.chat_widget.send_button.clicked.connect(self.send_message)

        logger.info("UI initialized")

    def create_menu_bar(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        select_schema_dir_action = QAction("Select Schema Directory...", self)
        select_schema_dir_action.triggered.connect(self.select_schema_directory)
        file_menu.addAction(select_schema_dir_action)

        reload_schemas_action = QAction("Reload Schemas", self)
        reload_schemas_action.setShortcut(QKeySequence("Ctrl+Shift+R"))
        reload_schemas_action.triggered.connect(self.reload_schemas)
        file_menu.addAction(reload_schemas_action)

        file_menu.addSeparator()

        clear_action = QAction("Clear Conversation", self)
        clear_action.setShortcut(QKeySequence("Ctrl+L"))
        clear_action.triggered.connect(self.clear_conversation)
        file_menu.addAction(clear_action)

        file_menu.addSeparator()

        reload_keys_action = QAction("Reload API Keys", self)
        reload_keys_action.setShortcut(QKeySequence("Ctrl+R"))
        reload_keys_action.triggered.connect(self.reload_api_keys)
        file_menu.addAction(reload_keys_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Provider menu
        self.provider_menu = menubar.addMenu("Provider")
        self.provider_menu.setEnabled(False)  # Disabled until providers are loaded

        # Settings menu
        settings_menu = menubar.addMenu("Settings")

        api_key_action = QAction("Set API Key...", self)
        api_key_action.setShortcut(QKeySequence("Ctrl+K"))
        api_key_action.triggered.connect(self.set_api_key)
        settings_menu.addAction(api_key_action)

        settings_menu.addSeparator()

        system_message_action = QAction("Set System Message...", self)
        system_message_action.setShortcut(QKeySequence("Ctrl+M"))
        system_message_action.triggered.connect(self.set_system_message)
        settings_menu.addAction(system_message_action)

        settings_menu.addSeparator()

        view_keys_action = QAction("View API Key Status...", self)
        view_keys_action.triggered.connect(self.view_api_key_status)
        settings_menu.addAction(view_keys_action)

        settings_menu.addSeparator()

        debug_action = QAction("Show Debug Info...", self)
        debug_action.setShortcut(QKeySequence("Ctrl+D"))
        debug_action.triggered.connect(self.show_debug_info)
        settings_menu.addAction(debug_action)

        # Help menu
        help_menu = menubar.addMenu("Help")

        about_action = QAction("About PyHyni", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def set_system_message(self):
        """Set the system message for the conversation"""
        if not self.current_provider:
            QMessageBox.warning(self, "No Provider Selected",
                              "Please select a provider first.")
            return

        provider_info = self.providers[self.current_provider]
        if not provider_info.supports_system_messages:
            QMessageBox.warning(self, "Not Supported",
                              f"{self.current_provider} does not support system messages.")
            return

        dialog = SystemMessageDialog(self.system_message, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.system_message = dialog.get_system_message()
            if self.chat_api and self.system_message:
                self.chat_api.context.set_system_message(self.system_message)
                logger.info(f"Set system message: {self.system_message[:50]}...")
                self.chat_widget.append_message("system", f"System message set: {self.system_message}")
            elif not self.system_message:
                if self.chat_api:
                    self.chat_api.context.clear_system_message()
                logger.info("Cleared system message")
                self.chat_widget.append_message("system", "System message cleared")

    def show_debug_info(self):
        """Show debug information dialog"""
        debug_info = "=== PyHyni Debug Information ===\n\n"

        # Environment variables
        debug_info += "Environment Variables (API Keys):\n"
        for key in ['OA_API_KEY', 'CL_API_KEY', 'DS_API_KEY', 'MS_API_KEY']:
            value = os.environ.get(key, '')
            if value:
                masked = value[:4] + "..." + value[-4:] if len(value) > 8 else "***"
                debug_info += f"  {key} = {masked}\n"
            else:
                debug_info += f"  {key} = <not set>\n"

        debug_info += "\n"

        # System message
        if self.system_message:
            debug_info += f"System Message: {self.system_message[:100]}...\n\n"
        else:
            debug_info += "System Message: <not set>\n\n"

        # Loaded providers
        debug_info += "Loaded Providers:\n"
        for display_name, provider_info in self.providers.items():
            debug_info += f"\n  {display_name}:\n"
            debug_info += f"    Schema name: {provider_info.name}\n"
            debug_info += f"    Endpoint: {provider_info.endpoint}\n"
            debug_info += f"    Expected env var: {get_env_var_name(provider_info.name)}\n"
            debug_info += f"    Supports system messages: {provider_info.supports_system_messages}\n"
            debug_info += f"    Supports streaming: {provider_info.supports_streaming}\n"

            if display_name in self.api_keys:
                key = self.api_keys[display_name]
                masked = key[:4] + "..." + key[-4:] if len(key) > 8 else "***"
                source = self.api_key_sources.get(display_name, "unknown")
                debug_info += f"    API Key: {masked} (from {source})\n"
            else:
                debug_info += f"    API Key: <not loaded>\n"

        debug_info += "\n"

        # Current state
        if self.current_provider and self.chat_api:
            debug_info += f"Current Provider: {self.current_provider}\n"
            debug_info += f"Current Model: {self.current_model}\n"
            debug_info += f"Multi-turn enabled: {self.chat_widget.is_multiturn_enabled()}\n"
            debug_info += f"Markdown enabled: {self.chat_widget.is_markdown_enabled()}\n"

            # Message history
            messages = self.chat_api.context.get_messages()
            debug_info += f"\nConversation History ({len(messages)} messages):\n"
            for i, msg in enumerate(messages):
                role = msg.get("role", "unknown")
                content = str(msg.get("content", ""))
                if isinstance(content, list) and content:
                    content = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
                debug_info += f"  {i+1}. {role}: {content[:50]}...\n"

        dialog = DebugDialog(self)
        dialog.set_debug_info(debug_info)
        dialog.exec()

    def select_schema_directory(self):
        """Let user select a schema directory"""
        logger.info("Opening schema directory selection dialog")

        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Schema Directory",
            self.schema_dir,
            QFileDialog.Option.ShowDirsOnly
        )

        if dir_path:
            logger.info(f"User selected schema directory: {dir_path}")
            self.schema_dir = dir_path
            self.settings.setValue("schema_dir", dir_path)
            self.load_schemas_from_directory(dir_path)

    def load_schemas_from_directory(self, directory: str):
        """Load all schemas from a directory"""
        logger.info(f"Loading schemas from directory: {directory}")

        # Clear existing providers
        self.providers.clear()
        self.provider_menu.clear()
        self.provider_menu.setEnabled(False)
        self.model_combo.clear()

        # Show progress dialog
        progress = QProgressDialog("Loading schemas...", "Cancel", 0, 0, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()

        # Create and start schema loader
        self.schema_loader = SchemaLoader(directory)
        self.schema_loader.provider_loaded.connect(self.on_provider_loaded)
        self.schema_loader.error_occurred.connect(self.on_schema_error)
        self.schema_loader.finished_loading.connect(lambda: self.on_schemas_loaded(progress))
        self.schema_loader.start()

    def on_provider_loaded(self, provider_info: ProviderInfo):
        """Handle a loaded provider"""
        logger.info(f"Provider loaded: {provider_info.display_name}")
        self.providers[provider_info.display_name] = provider_info

        # Load API key for this provider
        api_key = get_api_key_for_provider(provider_info.name)
        if api_key:
            self.api_keys[provider_info.display_name] = api_key
            # Determine source
            env_var = get_env_var_name(provider_info.name)
            if os.environ.get(env_var):
                self.api_key_sources[provider_info.display_name] = "environment"
            else:
                self.api_key_sources[provider_info.display_name] = ".hynirc"

    def on_schema_error(self, error_message: str):
        """Handle schema loading error"""
        logger.error(f"Schema loading error: {error_message}")
        self.chat_widget.append_message("error", error_message)

    def on_schemas_loaded(self, progress_dialog):
        """Handle completion of schema loading"""
        progress_dialog.close()

        logger.info(f"Schema loading completed. Loaded {len(self.providers)} providers")

        if not self.providers:
            self.show_no_schemas_message()
            return

        # Populate provider menu
        self.provider_menu.setEnabled(True)

        for display_name in sorted(self.providers.keys()):
            provider_info = self.providers[display_name]
            action = QAction(display_name, self)
            action.setCheckable(True)

            # Add checkmark if API key is available
            if display_name in self.api_keys:
                action.setText(f"{display_name} ✓")

            action.triggered.connect(lambda checked, p=display_name: self.change_provider(p))
            self.provider_menu.addAction(action)

        # Select first provider if available
        if self.providers:
            first_provider = sorted(self.providers.keys())[0]
            self.setup_provider(first_provider)

        # Show status
        self.chat_widget.append_message(
            "system",
            f"Loaded {len(self.providers)} provider(s) from {self.schema_dir}"
        )

        # Show API key status
        if self.api_keys:
            providers_with_keys = []
            for provider, source in self.api_key_sources.items():
                providers_with_keys.append(f"{provider} ({source})")

            message = f"API keys found for: {', '.join(providers_with_keys)}"
            self.chat_widget.append_message("system", message)

    def show_no_schemas_message(self):
        """Show message when no schemas are found"""
        logger.warning("No schema files found")
        self.chat_widget.append_message(
            "system",
            "No schema files found. Please select a directory containing schema JSON files "
            "via File → Select Schema Directory."
        )

    def reload_schemas(self):
        """Reload schemas from current directory"""
        logger.info("Reloading schemas")
        if os.path.exists(self.schema_dir):
            self.load_schemas_from_directory(self.schema_dir)
        else:
            QMessageBox.warning(
                self,
                "Schema Directory Not Found",
                f"Schema directory not found: {self.schema_dir}\n\n"
                "Please select a valid schema directory."
            )

    def setup_provider(self, display_name: str):
        """Setup a provider by display name"""
        if display_name not in self.providers:
            logger.error(f"Provider not found: {display_name}")
            return

        logger.info(f"Setting up provider: {display_name}")

        try:
            provider_info = self.providers[display_name]

            # Create context config - streaming OFF by default
            config = ContextConfig(
                enable_streaming_support=False,  # Disabled by default
                enable_validation=True,
                default_temperature=0.7,
                default_max_tokens=2000
            )

            logger.debug(f"Created context config with streaming_support={config.enable_streaming_support}")

            # Build chat API
            builder = ChatApiBuilder().schema(provider_info.schema_path).config(config)

            # Set API key if available
            if display_name in self.api_keys:
                builder.api_key(self.api_keys[display_name])
                logger.debug(f"Set API key for {display_name}")
            else:
                logger.warning(f"No API key available for {display_name}")

            self.chat_api = builder.build()

            # Set system message if available
            if self.system_message and provider_info.supports_system_messages:
                self.chat_api.context.set_system_message(self.system_message)
                logger.debug(f"Set system message for {display_name}")

            # Update model list
            self.model_combo.clear()
            self.model_combo.addItems(provider_info.available_models)
            if provider_info.default_model:
                self.model_combo.setCurrentText(provider_info.default_model)
                self.chat_api.context.set_model(provider_info.default_model)
                logger.debug(f"Set model to: {provider_info.default_model}")

            self.current_provider = display_name
            self.update_provider_menu()
            self.update_api_key_status()
            self.status_label.setText(f"Provider: {display_name}")

            # Update streaming checkbox state
            self.chat_widget.streaming_checkbox.setEnabled(provider_info.supports_streaming)
            if not provider_info.supports_streaming:
                self.chat_widget.streaming_checkbox.setChecked(False)
                self.chat_widget.streaming_checkbox.setToolTip("Streaming not supported by this provider")
            else:
                self.chat_widget.streaming_checkbox.setToolTip("Enable streaming responses")

            logger.info(f"Provider {display_name} setup completed")

        except Exception as e:
            logger.error(f"Failed to setup provider {display_name}: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to setup provider: {str(e)}")

    def change_provider(self, display_name: str):
        """Change the current provider"""
        if display_name != self.current_provider:
            logger.info(f"Changing provider from {self.current_provider} to {display_name}")

            # Cancel any ongoing operations
            self.cancel_current_operation()

            # Check if API key is set for the new provider
            if display_name not in self.api_keys:
                provider_info = self.providers[display_name]
                env_var = get_env_var_name(provider_info.name)
                reply = QMessageBox.question(
                    self,
                    "API Key Required",
                    f"No API key found for {display_name}.\n\n"
                    f"You can set it:\n"
                    f"1. Via environment variable ({env_var})\n"
                    f"2. In ~/.hynirc file\n"
                    f"3. Manually now\n\n"
                    f"Would you like to set it manually now?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    if not self.set_api_key_for_provider(display_name):
                        return
                else:
                    return

            self.setup_provider(display_name)

    def update_provider_menu(self):
        """Update the provider menu to reflect current selection and API key status"""
        for action in self.provider_menu.actions():
            display_name = action.text().replace(" ✓", "")

            # Update checkmark for API key availability
            if display_name in self.api_keys:
                action.setText(f"{display_name} ✓")
            else:
                action.setText(display_name)

            # Update checked state
            action.setChecked(display_name == self.current_provider)

    def update_api_key_status(self):
        """Update the API key status widget"""
        if self.current_provider and self.current_provider in self.api_keys:
            source = self.api_key_sources.get(self.current_provider, "manual")
            self.api_key_status.update_status(True, source)
        else:
            self.api_key_status.update_status(False)

    def reload_api_keys(self):
        """Reload API keys from environment"""
        logger.info("Reloading API keys")

        # Reload keys for all loaded providers
        for display_name, provider_info in self.providers.items():
            api_key = get_api_key_for_provider(provider_info.name)
            if api_key:
                self.api_keys[display_name] = api_key
                # Determine source
                env_var = get_env_var_name(provider_info.name)
                if os.environ.get(env_var):
                    self.api_key_sources[display_name] = "environment"
                else:
                    self.api_key_sources[display_name] = ".hynirc"

        self.update_provider_menu()
        self.update_api_key_status()

        # Update current provider if key was loaded
        if self.current_provider and self.current_provider in self.api_keys and self.chat_api:
            self.chat_api.context.set_api_key(self.api_keys[self.current_provider])

        QMessageBox.information(self, "API Keys Reloaded",
                              f"Found API keys for {len(self.api_keys)} provider(s)")

    def view_api_key_status(self):
        """Show detailed API key status"""
        status_text = "API Key Status:\n\n"

        for display_name in sorted(self.providers.keys()):
            provider_info = self.providers[display_name]
            env_var = get_env_var_name(provider_info.name)

            if display_name in self.api_keys:
                source = self.api_key_sources.get(display_name, "manual")
                # Show masked key
                key = self.api_keys[display_name]
                masked_key = key[:4] + "..." + key[-4:] if len(key) > 8 else "***"
                status_text += f"✓ {display_name}: {masked_key} (from {source})\n"
            else:
                status_text += f"✗ {display_name}: Not set\n"
                status_text += f"   Set via: {env_var}\n"

        QMessageBox.information(self, "API Key Status", status_text)

    def on_model_changed(self, model_name: str):
        """Handle model selection change"""
        if model_name and self.chat_api:
            logger.info(f"Changing model to: {model_name}")
            self.current_model = model_name
            self.chat_api.context.set_model(model_name)

    def set_api_key(self):
        """Set API key for current provider"""
        if self.current_provider:
            self.set_api_key_for_provider(self.current_provider)
        else:
            QMessageBox.warning(self, "No Provider Selected",
                              "Please select a provider first.")

    def set_api_key_for_provider(self, display_name: str) -> bool:
        """Set API key for a specific provider"""
        if display_name not in self.providers:
            return False

        provider_info = self.providers[display_name]
        current_key = self.api_keys.get(display_name, "")

        # Show current status in dialog
        if current_key:
            source = self.api_key_sources.get(display_name, "manual")
            masked_key = current_key[:4] + "..." + current_key[-4:] if len(current_key) > 8 else "***"
            prompt = f"Current key ({source}): {masked_key}\n\nEnter new API key for {display_name}:"
        else:
            prompt = f"Enter API key for {display_name}:"

        key, ok = QInputDialog.getText(
            self,
            f"Set API Key for {display_name}",
            prompt,
            echo=QInputDialog.EchoMode.Password
        )

        if ok and key:
            logger.info(f"Setting manual API key for {display_name}")
            self.api_keys[display_name] = key
            self.api_key_sources[display_name] = "manual"

            if display_name == self.current_provider and self.chat_api:
                self.chat_api.context.set_api_key(key)

            self.update_provider_menu()
            self.update_api_key_status()
            return True
        return False

    def send_message(self):
        """Send a message to the LLM"""
        message = self.chat_widget.get_input_text()
        if not message:
            return

        logger.info(f"Sending message: {message[:50]}...")

        if not self.current_provider:
            QMessageBox.warning(self, "No Provider Selected",
                              "Please select a provider from the Provider menu.")
            return

        # Check if API key is set
        if self.current_provider not in self.api_keys:
            provider_info = self.providers[self.current_provider]
            env_var = get_env_var_name(provider_info.name)
            QMessageBox.warning(
                self,
                "API Key Required",
                f"Please set an API key for {self.current_provider}.\n\n"
                f"You can:\n"
                f"1. Set it via Settings → Set API Key\n"
                f"2. Configure environment variable ({env_var})\n"
                f"3. Add it to ~/.hynirc file"
            )
            return

        # Handle multi-turn conversation
        keep_history = self.chat_widget.is_multiturn_enabled()
        if not keep_history:
            # Clear conversation history if multi-turn is disabled
            self.chat_api.context.clear_user_messages()
            logger.info("Cleared conversation history (multi-turn disabled)")

        # Disable send button during processing
        self.chat_widget.send_button.setEnabled(False)
        self.chat_widget.send_button.setText("Sending...")

        # Add user message to display
        self.chat_widget.append_message("user", message)

        # Check if streaming is requested and supported
        provider_info = self.providers[self.current_provider]
        use_streaming = self.chat_widget.is_streaming_enabled() and provider_info.supports_streaming

        logger.info(f"Streaming requested: {self.chat_widget.is_streaming_enabled()}, "
                   f"Provider supports: {provider_info.supports_streaming}, "
                   f"Will use streaming: {use_streaming}")

        if use_streaming:
            self.send_streaming_message(message, keep_history)
        else:
            self.send_non_streaming_message(message, keep_history)

    def send_streaming_message(self, message: str, keep_history: bool):
        """Send a message with streaming response"""
        logger.info(f"Sending message with streaming (keep_history={keep_history})")

        # Cancel any existing worker
        self.cancel_current_operation()

        # Create and start streaming worker
        self.streaming_worker = StreamingWorker(self.chat_api, message, keep_history)
        self.streaming_worker.chunk_received.connect(
            lambda chunk: self.on_streaming_chunk(chunk, self.current_model)
        )
        self.streaming_worker.error_occurred.connect(self.on_error)
        self.streaming_worker.finished_streaming.connect(self.on_streaming_finished)
        self.streaming_worker.finished.connect(self.on_worker_finished)
        self.streaming_worker.start()

        # Update button
        self.chat_widget.send_button.setText("Stop")
        self.chat_widget.send_button.setEnabled(True)
        self.chat_widget.send_button.clicked.disconnect()
        self.chat_widget.send_button.clicked.connect(self.cancel_current_operation)

    def send_non_streaming_message(self, message: str, keep_history: bool):
        """Send a message without streaming"""
        logger.info(f"Sending message without streaming (keep_history={keep_history})")

        # Cancel any existing worker
        self.cancel_current_operation()

        # Create and start non-streaming worker
        self.non_streaming_worker = NonStreamingWorker(self.chat_api, message, keep_history)
        self.non_streaming_worker.response_received.connect(
            lambda response: self.on_response_received(response, self.current_model)
        )
        self.non_streaming_worker.error_occurred.connect(self.on_error)
        self.non_streaming_worker.finished.connect(self.on_worker_finished)
        self.non_streaming_worker.start()

        # Update button
        self.chat_widget.send_button.setText("Cancel")
        self.chat_widget.send_button.setEnabled(True)
        self.chat_widget.send_button.clicked.disconnect()
        self.chat_widget.send_button.clicked.connect(self.cancel_current_operation)

    def on_streaming_chunk(self, chunk: str, model_name: str):
        """Handle a streaming chunk"""
        self.chat_widget.append_streaming_chunk(chunk, model_name)

    def on_streaming_finished(self):
        """Handle streaming completion"""
        logger.info("Streaming finished")
        self.chat_widget.finish_streaming_response()

    def on_response_received(self, response: str, model_name: str):
        """Handle a complete response"""
        logger.info("Response received")
        self.chat_widget.append_message("assistant", response, model_name)

    def on_error(self, error_message: str):
        """Handle an error"""
        logger.error(f"Error in response: {error_message}")
        self.chat_widget.append_message("error", error_message)

    def on_worker_finished(self):
        """Handle worker thread completion"""
        logger.info("Worker finished")

        # Reset button
        self.chat_widget.send_button.setText("Send")
        self.chat_widget.send_button.setEnabled(True)
        self.chat_widget.send_button.clicked.disconnect()
        self.chat_widget.send_button.clicked.connect(self.send_message)

        # Clean up workers
        if self.streaming_worker:
            self.streaming_worker.deleteLater()
            self.streaming_worker = None
        if self.non_streaming_worker:
            self.non_streaming_worker.deleteLater()
            self.non_streaming_worker = None

    def cancel_current_operation(self):
        """Cancel any ongoing operation"""
        logger.info("Cancelling current operation")

        if self.streaming_worker and self.streaming_worker.isRunning():
            self.streaming_worker.cancel()
            self.streaming_worker.wait()
        if self.non_streaming_worker and self.non_streaming_worker.isRunning():
            self.non_streaming_worker.cancel()
            self.non_streaming_worker.wait()

    def clear_conversation(self):
        """Clear the conversation"""
        reply = QMessageBox.question(
            self,
            "Clear Conversation",
            "Are you sure you want to clear the conversation?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            logger.info("Clearing conversation")
            self.chat_widget.clear_conversation()
            if self.chat_api:
                self.chat_api.context.clear_user_messages()
                # Re-apply system message if set
                if self.system_message:
                    provider_info = self.providers[self.current_provider]
                    if provider_info.supports_system_messages:
                        self.chat_api.context.set_system_message(self.system_message)

    def show_about(self):
        """Show about dialog"""
        about_text = "PyHyni - Python Hyni LLM Interface\n\n"
        about_text += "A PyQt6-based GUI for interacting with various Language Models.\n\n"

        if self.providers:
            about_text += "Loaded Providers:\n"
            for display_name, info in sorted(self.providers.items()):
                about_text += f"• {display_name} (v{info.version})\n"
                features = []
                if info.supports_streaming:
                    features.append("streaming")
                if info.supports_vision:
                    features.append("vision")
                if info.supports_system_messages:
                    features.append("system messages")
                if features:
                    about_text += f"  Supports: {', '.join(features)}\n"

        about_text += "\nFeatures:\n"
        about_text += "• Dynamic provider loading from schema files\n"
        about_text += "• Automatic API key loading from environment\n"
        about_text += "• Support for ~/.hynirc configuration\n"
        about_text += "• Markdown rendering for responses\n"
        about_text += "• Optional streaming responses\n"
        about_text += "• Multi-turn conversation support\n"
        about_text += "• System message configuration\n"
        about_text += "• Multiple model selection per provider\n\n"

        about_text += "API Key Environment Variables:\n"
        about_text += "• OA_API_KEY (OpenAI)\n"
        about_text += "• CL_API_KEY (Claude)\n"
        about_text += "• DS_API_KEY (DeepSeek)\n"
        about_text += "• MS_API_KEY (Mistral)\n\n"

        about_text += "Version: 1.0.0\n"
        about_text += "© 2024 PyHyni Project"

        QMessageBox.about(self, "About PyHyni", about_text)

    def closeEvent(self, event):
        """Handle window close event"""
        logger.info("PyHyni closing")

        # Cancel any ongoing operations
        self.cancel_current_operation()

        # Save settings
        self.settings.setValue("schema_dir", self.schema_dir)

        event.accept()


def main():
    app = QApplication(sys.argv)

    # Set application info
    app.setApplicationName("PyHyni")
    app.setOrganizationName("PyHyni")

    # Set application style
    app.setStyle("Fusion")

    # Create and show main window
    window = PyHyniMainWindow()
    window.show()

    logger.info("PyHyni started")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
