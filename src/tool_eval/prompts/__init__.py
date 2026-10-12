"""Versioned prompt loading. Prompts live as Markdown files next to this module."""

import hashlib
import re
from importlib.resources import files
from string import Template

from pydantic import BaseModel, ConfigDict

from tool_eval.exceptions import PromptNotFoundError

_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


class Prompt(BaseModel):
    """A prompt template together with the identifiers needed to reproduce a run."""

    model_config = ConfigDict(frozen=True)

    name: str
    version: str
    text: str
    sha256: str

    def render(self, **values: str) -> str:
        """Substitute `$placeholders` in the template.

        Args:
            **values: Values for the placeholders.

        Returns:
            The rendered prompt text.
        """
        return Template(self.text).substitute(values)


def load_prompt(name: str, version: str) -> Prompt:
    """Load `<name>/<version>.md` from the prompts package.

    Args:
        name: Prompt family, for example `system`.
        version: Version label, for example `v1`.

    Returns:
        The prompt with its SHA-256 content hash.

    Raises:
        PromptNotFoundError: If the name or version is invalid or the file does not exist.
    """
    if not (_SAFE_NAME.fullmatch(name) and _SAFE_NAME.fullmatch(version)):
        msg = f"Invalid prompt identifier: {name!r}/{version!r}"
        raise PromptNotFoundError(msg)
    resource = files(__package__) / name / f"{version}.md"
    if not resource.is_file():
        msg = f"Prompt not found: {name}/{version}"
        raise PromptNotFoundError(msg)
    text = resource.read_text(encoding="utf-8")
    return Prompt(
        name=name,
        version=version,
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )
