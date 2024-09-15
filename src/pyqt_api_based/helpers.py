from PyQt6.QtWidgets import QPushButton, QLineEdit, QComboBox

class StyledButton(QPushButton):
    """
    A QPushButton with custom styling.
    """
    def __init__(self, text, color):
        super().__init__(text)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
                cursor: pointer;
            }}
            QPushButton:hover {{
                background-color: {color}DD;
            }}
            QPushButton:pressed {{
                background-color: {color}BB;
            }}
        """)

class StyledLineEdit(QLineEdit):
    """
    A QLineEdit with custom placeholder styling.
    """
    def __init__(self, placeholder):
        super().__init__()
        self.setPlaceholderText(placeholder)
        self.setStyleSheet("""
            QLineEdit {
                border: 2px solid #ddd;
                border-radius: 5px;
                padding: 5px;
                background-color: #f5f5f5;
                color: #333;
            }
            QLineEdit:focus {
                border-color: #3498db;
            }
            QLineEdit::placeholder {
                color: #555;  /* Darker color for better contrast */
            }
        """)

class StyledComboBox(QComboBox):
    """
    A QComboBox with custom styling.
    """
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            QComboBox {
                border: 2px solid #ddd;
                border-radius: 5px;
                padding: 5px;
                background-color: #f5f5f5;
                color: #333;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 30px;
                border-left-width: 1px;
                border-left-color: #ddd;
                border-left-style: solid;
                border-top-right-radius: 5px;
                border-bottom-right-radius: 5px;
                background-color: #e0e0e0;
            }
            QComboBox::down-arrow {
                image: url(assets/icons/down_arrow.png);  /* Ensure to have this icon in the working directory */
            }
        """)