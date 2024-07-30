from reactagent.users.user import User
import json

class ConsoleUser(User):
    def interact(self, info):
        for k, v in info.items():
            v = User.indent_text(json.dumps(v,separators=(',\n', ': ')), 4)
            print(f"{k}:\n{v}")

        feedback = input(f"Please provide feedback based on the history, response entries, and observation, and questions: ")

        return feedback
