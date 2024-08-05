from reactagent.users.user import User
import json

class ConsoleUser(User):
    def interact(self, info):
        for k, v in info.items():
            if isinstance(v, dict):
                tempv = v
                v = ""
                for k2, v2 in tempv.items():
                    v += f"{k2}:\n  {v2}\n"
            v = User.indent_text(v, 2)
            print('-' * 64)
            print(f"{k}:\n{v}")

        print('*' * 64)
        feedback = input(f"Please provide feedback based on the history, response entries, and observation, and questions: ")

        return feedback
