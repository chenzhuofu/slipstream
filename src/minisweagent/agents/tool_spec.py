"""Tool specifications for mini-swe-agent, mirroring FoldAgent's tool_spec.py structure."""


def convert_tools_to_description(tools: list[dict]) -> str:
    ret = ""
    for i, tool in enumerate(tools):
        assert tool["type"] == "function"
        fn = tool["function"]
        if i > 0:
            ret += "\n"
        ret += f"---- BEGIN FUNCTION #{i + 1}: {fn['name']} ----\n"
        ret += f"Description: {fn['description']}\n"
        if "parameters" in fn:
            ret += "Parameters:\n"
            properties = fn["parameters"].get("properties", {})
            required_params = set(fn["parameters"].get("required", []))
            for j, (param_name, param_info) in enumerate(properties.items()):
                is_required = param_name in required_params
                param_status = "required" if is_required else "optional"
                param_type = param_info.get("type", "string")
                desc = param_info.get("description", "No description provided")
                if "enum" in param_info:
                    enum_values = ", ".join(f"`{v}`" for v in param_info["enum"])
                    desc += f"\nAllowed values: [{enum_values}]"
                ret += f"  ({j + 1}) {param_name} ({param_type}, {param_status}): {desc}\n"
        else:
            ret += "No parameters are required for this function.\n"
        ret += f"---- END FUNCTION #{i + 1} ----\n"
    return ret


def execute_bash_tool() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "execute_bash",
                "description": (
                    "Execute a bash command in the terminal.\n"
                    "* Long running commands: For commands that may run indefinitely, it should be run in the "
                    "background and the output should be redirected to a file, e.g. "
                    "`command = python3 app.py > server.log 2>&1 &`.\n"
                    "* One command at a time: You can only execute one bash command at a time. "
                    "If you need to run multiple commands sequentially, you can use `&&` or `;` to chain them together."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": (
                                "The bash command to execute. Can be empty string to view additional logs when "
                                "previous exit code is `-1`. Can be `C-c` (Ctrl+C) to interrupt the currently "
                                "running process. Note: You can only execute one bash command at a time. "
                                "If you need to run multiple commands sequentially, use `&&` or `;`."
                            ),
                        },
                    },
                    "required": ["command"],
                },
            },
        }
    ]


def code_summary_tool() -> list[dict]:
    """Summary tool for code tasks — included in the tool spec when context summarization is enabled.

    Parameters mirror the sections of SUMMARY_PROMPT_CODE from FoldAgent.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "summary",
                "description": (
                    "Called ONLY when told that the context is full. "
                    "Stop all work immediately and write a comprehensive handover for the next agent. "
                    "This summary will be the SOLE context for the next session — make it information-dense."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_context": {
                            "type": "string",
                            "description": "Preserve essential user requirements, goals, and clarifications in concise form.",
                        },
                        "completed": {
                            "type": "string",
                            "description": "Tasks completed so far, with brief results.",
                        },
                        "pending": {
                            "type": "string",
                            "description": "Tasks that still need to be done.",
                        },
                        "current_state": {
                            "type": "string",
                            "description": "Current variables, data structures, or relevant state.",
                        },
                        "code_state": {
                            "type": "string",
                            "description": "File paths, function signatures, data structures.",
                        },
                        "tests": {
                            "type": "string",
                            "description": "Failing cases, error messages, outputs.",
                        },
                        "changes": {
                            "type": "string",
                            "description": "Code edits, variable updates.",
                        },
                        "deps": {
                            "type": "string",
                            "description": "Dependencies, imports, external calls.",
                        },
                    },
                    "required": ["user_context", "completed", "pending", "current_state",
                                 "code_state", "tests", "changes", "deps"],
                },
            },
        }
    ]


TOOL_PROMPT = """\
You have access to the following functions:

{description}
If you choose to call a function ONLY reply in the following format with NO suffix:

<function=example_function_name>
<parameter=example_parameter_1>value_1</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Required parameters MUST be specified
- Only call one function at a time
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after.
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>"""
