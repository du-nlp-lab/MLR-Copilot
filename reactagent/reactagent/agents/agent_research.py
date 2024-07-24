""" This file contains the agent class for our AI research agent."""
import os
import sys
import anthropic
from reactagent.reactagent.llm import complete_text_fast, complete_text
from reactagent.schema import Action
from .agent import Agent
from .format import format_prompt_dict

from reactagent.prompt2model.prompt_parser import TaskType, PromptBasedInstructionParser

initial_prompt = """You are a helpful research assistant. You have access to the following tools:
{tools_prompt}

Research Problem: {task_description}

You do not know anything about this problem so far. 

Follow these instructions and do not forget them:
- First, come up with a high level plan based on your understanding of the problem and available tools and record it in the Research Plan and Status. You can revise the plan later.
- Research Plan and Status should well organized and succinctly keep track of 1) high level plan (can be revised), 2) what steps have been done and what steps are in progress, 3) short results and conclusions of each step after it has been performed. 
- Research Plan and Status must only include progress that has been made by previous steps. It should not include results not directly confirmed by the previous observation. 
- Performance numbers and estimates can only be confirmed and included in the status by running the code and observing the output.
- You should come up with a good experiment design that addresses the problem, and whenever applicable, define and measure the baseline performance of the relevant system or model before attempting any improvements.
- Follow the plan and try to achieve the goal as straightforwardly as possible.
- Highlight the supporting experiment results and reasoning before drawing any conclusions. 
- Do not try installing any new packages or libraries.
- If you believe you have solved the problem, you can use the Final Answer action to submit your answer. You can only submit once, so double check that you have achieved the goal before submitting.

Always respond in this format exactly:
{format_prompt}
Observation: 
```
the result of the action
```

"""

class ResearchAgent(Agent):
    """This class implements AI research agent with different configurations."""

    def __init__(self, args, env):
        super().__init__(args, env)
        self.valid_format_entires = format_prompt_dict.keys() # use all entries by default
        if args.valid_format_entires:
            self.valid_format_entires = args.valid_format_entires
        self.prompt_spec = PromptBasedInstructionParser(TaskType.TEXT_GENERATION)
        self.prompt_spec.parse_from_prompt(env.research_problem)
        task_desc = f'Instruction: {self.prompt_spec.instruction}\nExamples: {self.prompt_spec.examples}'
        self.initial_prompt = initial_prompt.format(tools_prompt=self.tools_prompt, tool_names=self.prompt_tool_names,  task_description=task_desc, format_prompt="\n".join([f"{k}: {format_prompt_dict[k]}" for k in self.valid_format_entires]))

        with open(os.path.join(self.log_dir , "main_log"), "a", 1) as f:
            f.write(self.initial_prompt + "\n")

    def run(self, env):
        while env.is_final() or len(self.history_steps) >= self.args.agent_max_steps:
            last_steps = self.args.max_steps_in_context
            last_observation_step = self.args.max_observation_steps_in_context
            curr_step = len(self.history_steps)

            # construct prompt for LLM based on previous steps

            prompt = self.initial_prompt
            if curr_step > last_steps:

                # retrieval action
                relevant_history = env.execute(Action("Retrieval from Research Log", {"current_plan": ""}))


                prompt += f"""
    Here is a summary of relevant actions and observations you have done:
    ```
    {relevant_history}
    ```
    Here are the exact several steps you have done most recently (up to 3 steps):
    """
            else:
                prompt += "\nNow let's start!\n\n"

            for idx in range(max(curr_step - last_steps, 0), curr_step):
                action_string = self.print_action(self.history_steps[idx]["action"], self.valid_format_entires)

                prompt += f"Action:\n{action_string}\nObservation:\n"
                if curr_step - idx > last_observation_step:
                    prompt += "<Done>\n\n"
                else:
                    try:
                        prompt += "```\n" + self.history_steps[idx]["observation"] + "\n```\n\n"
                    except:
                        import pdb; pdb.set_trace()

            # call LLM until the response is valid

            entries = None
            valid_response = False
            for _ in range(self.args.max_retries):
                log_file = os.path.join(self.log_dir , f"step_{curr_step}_log.log")
                completion = complete_text(prompt, self.args.llm_name)
                try:
                    entries = self.parse_entries(completion, self.valid_format_entires)
                    assert entries["Action"].strip() in self.all_tool_names
                    valid_response = True
                except:
                    print("Step", curr_step, file=sys.stderr)
                    print(anthropic.AI_PROMPT + "\n" + completion + "\nObservation:\n", file=sys.stderr)
                    print("Response is invalid and discarded", file=sys.stderr)
                    prompt += "\n\n Your response was in incorrect format. Please provide a valid response with all entries: " + ", ".join(self.valid_format_entires) + "\n\n"
                else:
                    break
            if not valid_response:
                return "No valid response after max_retries"

            # postprocess LLM output and parse to env actions

            rg = entries["Research Plan and Status"]
            action = entries["Action"].strip()
            raw_action_input = entries["Action Input"]

            new_research_plan_content = rg.strip("```") + "\n\n" 
            entries["Research Plan and Status"] = new_research_plan_content
            entries["Research Plan and Status"]= new_research_plan_content.replace("**", "")

            
            # parse the action input if we can ; other wise just return the original input and wait env to throw back an error
            parsing_error = ""
            try:
                action_input = self.parse_action_input(raw_action_input, self.action_infos[action])
                entries["Action Input"] = action_input
            except Exception as e:
                action_input = raw_action_input
                parsing_error = str(e)
                

            with open(os.path.join(self.log_dir , "main_log"), "a", 1) as f:
                f.write("Step " + str(curr_step) + ":\n")
                f.write(anthropic.AI_PROMPT + "\n" + self.print_action(entries, self.valid_format_entires) + "\nObservation:\n")


            # execute action in env

            if type(action_input) == dict:
                observation = env.execute(Action(action, action_input))
            else:
                # parsing failed, give agent parsing error
                usage = ",\n            ".join([f"{k}: [{v}]" for k, v in self.action_infos[action].usage.items()])
                usage = f"""{{
            {usage}
    }}"""
                invalid_action_error = f"The action input for {action} needs to be a valid json with proper entries. You may have missed the comma between entries or used triple quotes (json does not recognizes triple quotes). Please use the correct format and try again:\n{usage}"

                observation = "ActionInputParsingError: "+ parsing_error + "\n" + invalid_action_error

            # update history_steps

            # if observation is too long, we need to summarize it
            if len(observation) > 5000:
                log_file = os.path.join(self.log_dir , f"step_{curr_step}_summarize_observation_log.log")

                print("Observation is too long. Summarizing...", file=sys.stderr)
                observation = self.summarize_observation(self.print_action(entries, self.valid_format_entires), observation, log_file)

            info=dict(
                relevant_history=relevant_history,
                reflection=entries["Reflection"],
                research_plan_status=entries["Research Plan and Status"],
                fact_check=entries["Fact Check"],
                thought=entries["Thought"],
                questions=entries["Questions"],
                action=entries["Action"],
                action_input=entries["Action Input"],
                observation=entries["Observation"],
            )

            # give info to user and get feedback in return
            feedback = (yield info)

            self.history_steps.append({
                "step_idx": len(env.trace.steps),
                "action": entries,
                "observation": observation,
                "feedback": feedback,
            })

            # filter out ActionInputParsingError if last step is not action input parsing error
            if not observation.startswith("ActionInputParsingError"):
                self.history_steps = [step for step in self.history_steps if not step["observation"].startswith("ActionInputParsingError")]

            with open(os.path.join(self.log_dir , "main_log"), "a", 1) as f:
                f.write("\n```\n" + self.history_steps[-1]["observation"] + "\n```\n\n")


            # write to research log for retrieval
            summary_of_last_step = "Too long to summarize."
            for _ in range(self.args.max_retries):
                try:
                    summary_of_last_step = self.summarize_log_entry(
                        action=self.print_action(self.history_steps[-1]["action"], self.valid_format_entires),
                        observation=self.history_steps[-1]["observation"],
                        feedback=self.history_steps[-1]["feedback"],
                    )
                    break
                except Exception as e:
                    print(e)
                    print("Trying again.")

            env.execute(Action(
                name="Append Summary to Research Log",
                args={
                    "content": "\n\nStep " + str(curr_step) + ":\n" + summary_of_last_step + "\n"
                },
            ))

            step_idx = len(env.trace.steps) - 1
            self.save(os.path.join(self.log_dir , f"agent_{step_idx}_{curr_step}.json"))


    ################### Helper functions #####################

    def summarize_observation(self, action, observation, log_file, bs = 10000):
        """ Summarize the observation if it is too long with a sliding window of size bs """

        bs = 10000
        blocks = [observation[i:i+bs] for i in range(0, len(observation), bs)]
        descriptions = []
        for idx, b in enumerate(blocks):
            start_line_number = bs*idx+1
            end_line_number = bs*idx+1 + len(b)
            prompt = f"""
{action}

The full observation is too long. Given this (partial) observation from character {start_line_number} to character {end_line_number}: 
``` 
{b}
```
Summarize the observation concisely in this format:
[Observation]: Summarize all relevant details in the observation objectively

Do not include any result that is guessed rather than directly confirmed by the observation. Do not include additional information or suggestions.
"""

            completion = complete_text_fast(prompt)
            descriptions.append(completion)
        if len(descriptions) == 1:
            completion = descriptions[0]
        else:
            descriptions = "\n\n".join(["Segment {idx}: \n\n" + s for s in descriptions])
            prompt = f"""
{action}

The full observation is too long. 
Given summaries for each segments of the whole observation, summarize to get a cohesive description of the entire observation.
{descriptions}

Summarize the observation concisely in this format:
[Observation]: Summarize all relevant details in the observation objectively

Do not include any result that is guessed rather than directly confirmed by the observation. Do not include additional information or suggestions.
"""

            completion = complete_text_fast(prompt)
        try:
            return completion.split("[Observation]:")[1]
        except:
            return completion

    @staticmethod
    def summarize_log_entry(action, observation, feedback, **kwargs):
        """ Summarize the action and observation to an entry in the research log """

        prompt = f"""Given your action, the observation, and the human feedback: 
        [Action]:
        {action}
        [Observation]:
        ```
        {observation}
        ```
        [Feedback]:
        {feedback}
        
        Summarize your action and the observation in this format:
        [Reasoning]: Summarize the reasoning behind the action
        [Action]: Summarize all relevant details of the action objectively
        [Observation]: Summarize all relevant details in the observation objectively
        [Feedback]: Summarize all relevant details in the human feedback objectively
        Do not include any result that is guessed rather than directly confirmed by the observation. Do not include additional information or suggestions.
        """

        summary = "[Reasoning]:" + complete_text_fast(prompt).split("[Reasoning]:")[1]
        return summary
    
