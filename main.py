# Calculates confusion matrix for OCRs. Originally developed
# for Tesseract OCR, but more OCRs might be added in future.

import logging

import click

import commands
from utils.cmd_root import root
from errors import Error
import dataclasses
import importlib
import inspect
import pkgutil
from types import ModuleType
from typing import Callable, Dict, Iterable


LOG = logging.getLogger(__name__)


@dataclasses.dataclass
class CollectedCommand:
    name: str
    action_module: ModuleType
    process_function: Callable = None


CollectedCommandsType = Dict[str, CollectedCommand]


def collect_commands(commands_module) -> CollectedCommandsType:
    LOG.debug("Collecting commands...")
    discovered_actions = {}

    for m in pkgutil.iter_modules(commands_module.__path__):
        action_name = m.name
        module_str = ".".join([commands_module.__name__, m.name])
        LOG.debug(f"    '{action_name}'")
        action_module = importlib.import_module(module_str)

        module_commands: Iterable[click.Command] = inspect.getmembers(
            action_module, lambda obj: isinstance(obj, click.Command)
        )

        for cmd_name, cmd in module_commands:
            cmd_descr = CollectedCommand(
                cmd_name, action_module
            )
            root.add_command(cmd)
            discovered_actions[cmd_name] = cmd_descr

    return discovered_actions


def main():
    try:
        collect_commands(commands)
        root()
    except Error as e:
        LOG.error(f"Error: {e.message}")
        pass
