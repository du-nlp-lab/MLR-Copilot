from reactagent.users import User

class ConsoleUser(User):
    def interact(self, info):
        for k, v in info.items():
            print(f"{k}:\n{User.indent_text(v, 4)}")

        feedback = input(f"Please provide feedback based on the history, response entries, and observation, and questions: ")

        return feedback
