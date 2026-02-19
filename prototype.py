#!/usr/bin/env python3
"""
PC Build Advisor - Uses GPT-4o to recommend components within your budget.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

COMPONENTS = [
    "CPU",
    "CPU Cooler",
    "Motherboard",
    "Memory (RAM)",
    "Storage",
    "Video Card (GPU)",
    "Case",
    "Power Supply (PSU)",
    "Operating System",
    "Monitor",
]

def get_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        key = input("Enter your OpenAI API key: ").strip()
    return key


def prompt_user_inputs() -> tuple[dict[str, str], float]:
    print("\n" + "=" * 60)
    print("        🖥️  PC BUILD ADVISOR  🖥️")
    print("=" * 60)
    print("Let's gather your preferences for each component.")
    print("Leave a field blank if you have no preference.\n")

    preferences: dict[str, str] = {}

    for component in COMPONENTS:
        value = input(f"  {component}: ").strip()
        preferences[component] = value if value else "No preference"

    print()
    while True:
        budget_str = input("What is your total budget (USD)? $").strip()
        try:
            budget = float(budget_str.replace(",", ""))
            if budget <= 0:
                raise ValueError
            break
        except ValueError:
            print("   Please enter a valid positive number.")

    print()
    use_case = input(
        "What will you primarily use this PC for?\n"
        "   (e.g., gaming, video editing, 3D rendering, general use): "
    ).strip()
    if not use_case:
        use_case = "General use / gaming"

    preferences["_use_case"] = use_case

    return preferences, budget


def build_prompt(preferences: dict[str, str], budget: float) -> str:
    use_case = preferences.pop("_use_case", "General use")

    lines = []
    for component, pref in preferences.items():
        lines.append(f"  - {component}: {pref}")

    component_block = "\n".join(lines)

    return f"""You are an expert PC builder. A user wants to build a PC with the following specifications:

**Budget:** ${budget:,.2f} USD (total, all-inclusive)
**Primary Use Case:** {use_case}

**User Preferences / Constraints:**
{component_block}

Your task:
1. Recommend specific products for each component that maximize performance within the budget.
2. For each component, provide:
   - Recommended product name and model
   - Estimated price (USD)
   - Brief reason for the recommendation (1-2 sentences)
3. Ensure all components are compatible with each other (socket, chipset, wattage, form factor, etc.).
4. Provide a **Total Estimated Cost** at the end.
5. If the budget is tight, prioritize the components that most impact the user's use case.
6. Mention any potential upgrades or trade-offs the user should be aware of.
7. If the user left a component preference blank, choose the best value option.

Format your response clearly with each component as a section header."""


def get_recommendations(client: OpenAI, prompt: str) -> str:
    print("\nConsulting GPT-4o for your personalized build recommendations...\n")
    print("=" * 60)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a knowledgeable and friendly PC building expert. "
                    "You provide accurate, up-to-date hardware recommendations "
                    "with real product names and realistic pricing. "
                    "You always ensure component compatibility."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=2000,
    )

    return response.choices[0].message.content


def main():
    load_dotenv()
    api_key = get_api_key()
    client = OpenAI(api_key=api_key)

    preferences, budget = prompt_user_inputs()
    prompt = build_prompt(preferences, budget)
    recommendations = get_recommendations(client, prompt)

    print(recommendations)
    print("=" * 60)

    save = input("\nWould you like to save these recommendations to a file? (y/n): ").strip().lower()
    if save == "y":
        filename = "pc_build_recommendations.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write("PC BUILD RECOMMENDATIONS\n")
            f.write("=" * 60 + "\n\n")
            f.write(recommendations)
            f.write("\n\n" + "=" * 60 + "\n")
        print(f"Recommendations saved to '{filename}'")

    print("\nHappy building!\n")


if __name__ == "__main__":
    main()