"""
Example: Running Multiple Bots Simultaneously

This script demonstrates how to run multiple poker bots at the same time.
Useful for testing bot vs bot gameplay.
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from poker_client_enhanced import play_poker


async def run_multiple_bots():
    """Run multiple bots concurrently."""
    
    print("\n" + "="*70)
    print("🎰 RUNNING MULTIPLE POKER BOTS")
    print("="*70)
    
    # Configuration
    API_KEY = "dev"
    TABLE_ID = "table-1"
    SERVER_HOST = "localhost"
    SERVER_PORT = "8080"
    
    # Bot configurations
    bots = [
        {
            'player_id': 'bot1',
            'model_path': None,  # or path to trained model
        },
        {
            'player_id': 'bot2',
            'model_path': None,
        },
        # Add more bots as needed
        # {
        #     'player_id': 'bot3',
        #     'model_path': None,
        # },
    ]
    
    print(f"\nStarting {len(bots)} bots:")
    for i, bot in enumerate(bots, 1):
        print(f"  {i}. {bot['player_id']}")
    
    print("\n" + "="*70)
    print("⚠️  IMPORTANT: Make sure the Go server is running!")
    print("   Start it with: go run cmd/main.go")
    print("="*70)
    
    # Create tasks for each bot
    tasks = []
    for bot in bots:
        url = f"ws://{SERVER_HOST}:{SERVER_PORT}/ws?apiKey={API_KEY}&table={TABLE_ID}&player={bot['player_id']}"
        task = asyncio.create_task(
            play_poker(url, bot['player_id'], bot['model_path'])
        )
        tasks.append(task)
    
    # Run all bots concurrently
    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping all bots...")
        for task in tasks:
            task.cancel()
        print("✅ All bots stopped")


if __name__ == "__main__":
    print("\n🤖 Multiple Poker Bots Example")
    print("="*70)
    
    try:
        asyncio.run(run_multiple_bots())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
