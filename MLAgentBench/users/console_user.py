from MLAgentBench.users import User

class ConsoleUser(User):
    @staticmethod
    def interact(relevant_history, entries, observation):
        print(f"Relevant History:\n{User.indent_text(relevant_history, 4)}")

        print("Entries:")
        for k, v in entries.items():
            print(f"  {k}:\n{User.indent_text(v, 4)}")

        print(f"Observation:\n{User.indent_text(observation, 4)}")

        feedback = input("Please provide feedback based on the history, response entries, and observation: ")

        return feedback