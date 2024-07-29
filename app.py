import gradio as gr
from uuid import uuid4 as uuid
from pathlib import Path
from reactagent.environment import Environment
from reactagent.agents.agent_research import ResearchAgent
from reactagent.runner import create_parser

class SessionInfo:
    def __init__(self):
        self.coro_cache = {}
        self.parser = create_parser()

    def make_session(self, prompt, session_hash):
        id = session_hash

        rawargs = [
            '--research-problem', prompt,
            '--log-dir', str(Path('logs', id)),
            '--work-dir', str(Path('workspaces', id)),
        ]

        args = self.parser.parse_args(rawargs)

        env = Environment(args)
        agent = ResearchAgent(args, env)
        coro = agent.run(env)

        self.coro_cache[id] = coro
        return id

    def get_response(self, human_input, session_hash):
        coro_input = human_input
        if session_hash not in self.coro_cache:
            self.make_session(human_input, session_hash)
            coro_input = None

        try:
            output = self.coro_cache[session_hash].send(coro_input)
        except StopIteration:
            output = None
            del self.coro_cache[session_hash]

        return output

session_info = SessionInfo()

def predict(message, history, request: gr.Request):
    response = session_info.get_response(message, request.session_hash)
    if response is None:
        response = "Agent is finished. Enter a new instruction."
    return response

if __name__ == "__main__":
    demo = gr.ChatInterface(predict)
    demo.launch()
