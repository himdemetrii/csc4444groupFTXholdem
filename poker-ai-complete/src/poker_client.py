"""
WebSocket Poker Client - Integrates trained PyTorch model with Go game server
"""

import asyncio
import json
import websockets
import torch
import random
from typing import Dict, List, Optional, Tuple

# Import your existing model classes
from poker_model import SimplePokerNet, calculate_hand_strength


class PokerBot:
    """AI poker player that connects to the WebSocket server."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the bot with a trained model."""
        self.model = SimplePokerNet(embed_dim=64, hidden_dim=128, num_actions=7)
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
            print(f"✅ Loaded model from {model_path}")
        
        self.model.eval()
        
        # Action mapping: Model output -> Server action
        self.action_map = {
            0: None,  # Padding
            1: "FOLD",
            2: "CHECK",
            3: "CALL",
            4: "RAISE",  # Small raise
            5: "RAISE",  # Medium raise
            6: "RAISE",  # Large raise
        }
        
        # Raise amount mapping based on action type
        self.raise_amounts = {
            4: 2.0,   # Small: 2x BB
            5: 3.5,   # Medium: 3.5x BB
            6: 5.0,   # Large: 5x BB
        }
    
    def _parse_card(self, card_dict: Dict) -> Tuple[int, int]:
        """
        Convert server card format to model format.
        Server: {"rank": "A", "suit": "HEART"}
        Model: [rank_index, suit_index]
        """
        rank_map = {
            '2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6,
            '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12
        }
        
        suit_map = {
            'SPADE': 0, 'HEART': 1, 'DIAMOND': 2, 'CLUB': 3
        }
        
        rank = card_dict.get('rank', '2')
        suit = card_dict.get('suit', 'SPADE')
        
        return [rank_map.get(rank, 0), suit_map.get(suit, 0)]
    
    def _extract_my_cards(self, state: Dict, player_id: str) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Extract hole cards and board cards from game state.
        Returns: (hole_cards, board_cards) in model format
        """
        # Default empty cards
        hole_cards = [[0, 0], [0, 0]]
        board_cards = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
        
        try:
            table = state.get('table', {})
            
            # Find our player
            players = table.get('players', [])
            my_player = None
            for p in players:
                if p and p.get('id') == player_id:
                    my_player = p
                    break
            
            # Extract hole cards
            if my_player and 'cards' in my_player:
                cards = my_player['cards']
                if len(cards) >= 2:
                    hole_cards[0] = self._parse_card(cards[0])
                    hole_cards[1] = self._parse_card(cards[1])
            
            # Extract board cards
            board = state.get('board', [])
            for i, card in enumerate(board[:5]):
                if card:
                    board_cards[i] = self._parse_card(card)
        
        except Exception as e:
            print(f"⚠️  Error extracting cards: {e}")
        
        return hole_cards, board_cards
    
    def _is_my_turn(self, state: Dict, player_id: str) -> bool:
        """Check if it's our turn to act."""
        try:
            table = state.get('table', {})
            players = table.get('players', [])
            to_act_idx = state.get('toActIdx', -1)
            
            if to_act_idx < 0 or to_act_idx >= len(players):
                return False
            
            current_player = players[to_act_idx]
            return current_player and current_player.get('id') == player_id
        
        except Exception as e:
            print(f"⚠️  Error checking turn: {e}")
            return False
    
    def _get_to_call_amount(self, state: Dict, player_id: str) -> int:
        """
        Calculate how much the player needs to call.
        This is critical for determining legal actions.
        """
        try:
            # Note: The engine tracks roundBets internally, but it's not exposed in state
            # For now, we'll use a heuristic based on the phase and player actions
            # In a production system, the server should send toCall in the state
            
            # For now, return 0 (CHECK available) as default
            # This will be refined once we see actual game states
            return 0
        except Exception as e:
            print(f"⚠️  Error calculating to-call: {e}")
            return 0
    
    def _get_legal_actions(self, state: Dict, player_id: str) -> Dict[str, bool]:
        """
        Determine which actions are legal based on engine rules.
        
        Engine rules (from engine.go):
        - CHECK: Only when toCall == 0
        - CALL: When toCall > 0
        - RAISE: Must be >= MinRaise above current bet
        - FOLD: Always available
        """
        legal = {"CHECK": False, "CALL": False, "RAISE": True, "FOLD": True}
        
        try:
            table = state.get('table', {})
            players = table.get('players', [])
            
            # Find our player
            my_player = None
            for p in players:
                if p and p.get('id') == player_id:
                    my_player = p
                    break
            
            if not my_player:
                return legal
            
            my_chips = my_player.get('chips', 0)
            
            # Calculate to-call amount (simplified)
            to_call = self._get_to_call_amount(state, player_id)
            
            # Engine logic:
            # - If toCall == 0: CHECK is legal
            # - If toCall > 0: CALL is legal (if we have chips)
            # - RAISE is legal if we have enough chips
            
            if to_call == 0:
                legal["CHECK"] = True
                legal["CALL"] = False  # Can't call when there's no bet
            else:
                legal["CHECK"] = False
                legal["CALL"] = my_chips >= to_call
            
            # RAISE is legal if we have chips beyond the call amount
            # Engine requires raise >= MinRaise (typically BigBlind, updates per raise)
            legal["RAISE"] = my_chips > to_call
            
            # FOLD is always legal
            legal["FOLD"] = True
        
        except Exception as e:
            print(f"⚠️  Error determining legal actions: {e}")
        
        return legal
    
    def _map_action_to_server(self, model_action: int, state: Dict, 
                               player_id: str, bb: int = 10) -> Dict:
        """
        Convert model action to server message format.
        Respects engine validation rules from engine.go:
        - CHECK only when toCall == 0
        - CALL when toCall > 0
        - RAISE must be >= MinRaise (typically BB, resets per street)
        
        Returns: {"type": "act", "action": "CALL", "amount": 0}
        """
        legal = self._get_legal_actions(state, player_id)
        
        # Get my player info
        my_chips = 0
        table = state.get('table', {})
        for p in table.get('players', []):
            if p and p.get('id') == player_id:
                my_chips = p.get('chips', 0)
                break
        
        # Get base action
        action_str = self.action_map.get(model_action)
        
        if action_str is None:  # Padding action
            action_str = "CHECK" if legal["CHECK"] else "CALL"
        
        # Validate and correct illegal actions
        if action_str == "CHECK" and not legal["CHECK"]:
            # Can't check when there's a bet - must call or fold
            action_str = "CALL" if legal["CALL"] else "FOLD"
        
        if action_str == "CALL" and not legal["CALL"]:
            # Can't call when there's no bet - check instead
            if legal["CHECK"]:
                action_str = "CHECK"
            else:
                action_str = "FOLD"
        
        if action_str == "RAISE" and not legal["RAISE"]:
            # Can't raise - fall back to call or check
            if legal["CALL"]:
                action_str = "CALL"
            elif legal["CHECK"]:
                action_str = "CHECK"
            else:
                action_str = "FOLD"
        
        # Build message
        msg = {
            "type": "act",
            "action": action_str
        }
        
        # Add amount for raises
        # Engine rules: raise amount is the RAISE SIZE (not total bet)
        # Must be >= MinRaise (typically BB, updated per raise)
        if action_str == "RAISE":
            multiplier = self.raise_amounts.get(model_action, 2.5)
            raise_size = int(bb * multiplier)
            
            # Ensure we have enough chips for this raise
            # If not, reduce to max chips or convert to call
            to_call = self._get_to_call_amount(state, player_id)
            total_needed = to_call + raise_size
            
            if my_chips < total_needed:
                if my_chips <= to_call:
                    # Can only call (or fold)
                    msg["action"] = "CALL"
                elif my_chips > to_call:
                    # Short all-in raise
                    raise_size = my_chips - to_call
                    msg["amount"] = raise_size
                else:
                    # Can't raise, fold
                    msg["action"] = "FOLD"
            else:
                msg["amount"] = raise_size
        
        return msg
    
    def decide_action(self, state: Dict, player_id: str, bb: int = 10) -> Dict:
        """
        Use the trained model to decide what action to take.
        Returns: Server message dict
        """
        try:
            # Extract cards
            hole_cards, board_cards = self._extract_my_cards(state, player_id)
            
            # Convert to tensors
            hole_tensor = torch.tensor([hole_cards], dtype=torch.long)
            board_tensor = torch.tensor([board_cards], dtype=torch.long)
            
            # Get model prediction
            with torch.no_grad():
                logits, value = self.model(hole_tensor, board_tensor)
                policy = torch.softmax(logits, dim=-1)
                
                # Get top action
                action_idx = policy.argmax(dim=1).item()
                confidence = policy[0, action_idx].item()
            
            # Calculate hand strength for logging
            hand_str = calculate_hand_strength(hole_cards, board_cards)
            
            print(f"\n🤔 Decision:")
            print(f"   Hand Strength: {hand_str:.3f}")
            print(f"   Model Action: {self.action_map.get(action_idx, 'UNKNOWN')} (confidence: {confidence:.2%})")
            print(f"   Value Estimate: {value.item():.3f}")
            
            # Convert to server format
            action_msg = self._map_action_to_server(action_idx, state, player_id, bb)
            
            return action_msg
        
        except Exception as e:
            print(f"❌ Error deciding action: {e}")
            # Fallback to safe action
            return {"type": "act", "action": "CHECK"}


async def play_poker(url: str, player_id: str, model_path: Optional[str] = None):
    """
    Connect to the poker server and play using the trained model.
    
    Args:
        url: WebSocket URL (e.g., "ws://localhost:8080/ws?apiKey=dev&table=table-1&player=bot1")
        player_id: Your player ID
        model_path: Path to saved model weights (optional)
    """
    bot = PokerBot(model_path)
    
    print(f"\n🤖 Poker Bot Starting")
    print(f"   Player ID: {player_id}")
    print("="*60)
    
    try:
        async with websockets.connect(url) as ws:
            print(f"✅ Connected to {url}")
            
            # Send join message
            await ws.send(json.dumps({"type": "join"}))
            print(f"📨 Sent join request")
            
            # Main game loop
            while True:
                try:
                    # Receive state update
                    msg = await ws.recv()
                    data = json.loads(msg)
                    
                    msg_type = data.get('type')
                    
                    if msg_type == 'state':
                        state = data.get('state', {})
                        phase = state.get('phase', 'UNKNOWN')
                        hand = state.get('hand', 0)
                        pot = state.get('pot', 0)
                        
                        print(f"\n📊 State Update: Hand #{hand} | Phase: {phase} | Pot: {pot}")
                        
                        # Check if it's our turn
                        if bot._is_my_turn(state, player_id):
                            print("🎯 It's our turn!")
                            
                            # Decide action using model
                            action_msg = bot.decide_action(state, player_id)
                            
                            # Send action
                            await ws.send(json.dumps(action_msg))
                            print(f"✉️  Sent action: {action_msg}")
                    
                    elif msg_type == 'error':
                        error_msg = data.get('message', 'Unknown error')
                        print(f"❌ Server error: {error_msg}")
                    
                    else:
                        print(f"📥 Received: {msg_type}")
                
                except websockets.exceptions.ConnectionClosed:
                    print("\n🔌 Connection closed")
                    break
                
                except json.JSONDecodeError as e:
                    print(f"⚠️  JSON decode error: {e}")
                    continue
                
                except Exception as e:
                    print(f"⚠️  Error in game loop: {e}")
                    continue
    
    except Exception as e:
        print(f"❌ Connection error: {e}")


# Example usage
if __name__ == "__main__":
    # Configuration
    API_KEY = "dev"
    TABLE_ID = "table-1"
    PLAYER_ID = "bot1"
    SERVER_HOST = "localhost"
    SERVER_PORT = "8080"
    
    # Construct WebSocket URL
    url = f"ws://{SERVER_HOST}:{SERVER_PORT}/ws?apiKey={API_KEY}&table={TABLE_ID}&player={PLAYER_ID}"
    
    # Optional: path to saved model
    MODEL_PATH = None  # or "poker_model.pth" if you saved weights
    
    # Run the bot
    print("\n" + "="*60)
    print("🎰 POKER BOT CLIENT")
    print("="*60)
    
    asyncio.run(play_poker(url, PLAYER_ID, MODEL_PATH))
