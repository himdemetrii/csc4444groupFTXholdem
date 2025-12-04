"""
Enhanced WebSocket Poker Client - Uses game context for better decisions
"""

import asyncio
import json
import websockets
import torch
from typing import Dict, List, Optional, Tuple
import ssl

# Import the enhanced model
from poker_model_enhanced import EnhancedPokerNet, extract_context_features, calculate_hand_strength


class EnhancedPokerBot:
    """
    AI poker player that uses game context (chips, pot, phase, position).
    Much smarter than the basic version!
    """
    
    def __init__(self, model_path: Optional[str] = None, starting_stack: int = 1000):
        """Initialize the bot with an enhanced model."""
        self.model = EnhancedPokerNet(embed_dim=64, hidden_dim=256, num_actions=7)
        self.starting_stack = starting_stack
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
            print(f"✅ Loaded model from {model_path}")
        
        self.model.eval()
        
        # Action mapping
        self.action_map = {
            0: None,  # Padding
            1: "FOLD",
            2: "CHECK",
            3: "CALL",
            4: "RAISE",  # Small
            5: "RAISE",  # Medium
            6: "RAISE",  # Large
        }
        
        self.raise_amounts = {
            4: 2.0,   # Small: 2x BB
            5: 3.5,   # Medium: 3.5x BB
            6: 5.0,   # Large: 5x BB
        }
    
    def _parse_card(self, card_dict: Dict) -> Tuple[int, int]:
        """Convert server card format to model format."""
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
        """Extract hole cards and board cards."""
        hole_cards = [[0, 0], [0, 0]]
        board_cards = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
        
        try:
            table = state.get('table', {})
            players = table.get('players', [])
            
            # Find our player
            for p in players:
                if p and p.get('id') == player_id:
                    cards = p.get('cards', [])
                    if len(cards) >= 2:
                        hole_cards[0] = self._parse_card(cards[0])
                        hole_cards[1] = self._parse_card(cards[1])
                    break
            
            # Extract board
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
    
    def _get_legal_actions(self, state: Dict, player_id: str) -> Dict[str, bool]:
        """Return the server-reported legal actions."""
        legal = state.get("legalActions", {})

    # Always provide all 4 keys to avoid KeyErrors
        if not legal:
            return {
                "CHECK": True,
                "CALL":  True,
                "RAISE": True,
                "FOLD":  True,
            }
        
        return {
            "CHECK": legal.get("CHECK", False),
            "CALL":  legal.get("CALL", False),
            "RAISE": legal.get("RAISE", False),
            "FOLD":  legal.get("FOLD", True),
      }
    
    def _map_action_to_server(self, model_action: int, state: Dict, 
                               player_id: str, bb: int = 10) -> Dict:
        """Convert model action to server format."""
        legal = self._get_legal_actions(state, player_id)
        
        # Get my chips
        my_chips = 0
        table = state.get('table', {})
        for p in table.get('players', []):
            if p and p.get('id') == player_id:
                my_chips = p.get('chips', 0)
                break
        
        action_str = self.action_map.get(model_action)

        if action_str is None:
            action_str = "CHECK" if legal["CHECK"] else "CALL"
        
        # Validate
        if action_str == "CHECK" and not legal["CHECK"]:
            action_str = "CALL" if legal["CALL"] else "FOLD"
        
        if action_str == "CALL" and not legal["CALL"]:
            action_str = "CHECK" if legal["CHECK"] else "FOLD"
        
        msg = {
            "type": "act",
            "action": action_str
        }
        
        # Add raise amount
        if action_str == "RAISE":
            multiplier = self.raise_amounts.get(model_action, 2.5)
            raise_size = int(bb * multiplier)

            raw_legal = state.get("legalActions", {})
            raise_info = raw_legal.get("RAISE")

            min_raise = None
            max_raise = None

            if isinstance(raise_info, dict):
                min_raise = raise_info.get("min")
                max_raise = raise_info.get("max")

            if min_raise is not None:
                raise_size = max(raise_size, min_raise)
            if max_raise is not None:
                raise_size = min(raise_size, max_raise)
            
            # Check chip constraints
            if my_chips < raise_size:
                if legal.get("CALL", False):
                    msg["action"] = "CALL"
                    return msg
                
                msg["action"] = "FOLD"
                return msg
            
            msg["amount"] = raise_size
        
        return msg
    
    def decide_action(self, state: Dict, player_id: str, bb: int = 10) -> Dict:
        """
        Use the enhanced model to decide action.
        Now considers: chips, pot, phase, position, SPR!
        """
        try:
            # Extract cards
            hole_cards, board_cards = self._extract_my_cards(state, player_id)
            
            # Extract game context (THE NEW PART!)
            context_features = extract_context_features(state, player_id, self.starting_stack)
            
            # Convert to tensors
            hole_tensor = torch.tensor([hole_cards], dtype=torch.long)
            board_tensor = torch.tensor([board_cards], dtype=torch.long)
            context_tensor = torch.tensor([context_features], dtype=torch.float32)
            
            # Get model prediction WITH CONTEXT
            with torch.no_grad():
                logits, value = self.model(hole_tensor, board_tensor, context_tensor)
                policy = torch.softmax(logits, dim=-1)
                action_idx = policy.argmax(dim=1).item()
                confidence = policy[0, action_idx].item()
            
            # Calculate hand strength for logging
            hand_str = calculate_hand_strength(hole_cards, board_cards)
            
            # Pretty print context
            my_chips = context_features[0] * self.starting_stack
            pot = context_features[1] * self.starting_stack
            phase_names = ['WAITING', 'PREFLOP', 'FLOP', 'TURN', 'RIVER', 'SHOWDOWN']
            phase_idx = int(context_features[2] * 5)
            phase_name = phase_names[phase_idx] if 0 <= phase_idx < len(phase_names) else 'UNKNOWN'
            spr = context_features[5] * 10.0
            
            print(f"\n🤔 Enhanced Decision:")
            print(f"   Hand Strength: {hand_str:.3f}")
            print(f"   My Chips: {my_chips:.0f} | Pot: {pot:.0f} | SPR: {spr:.1f}")
            print(f"   Phase: {phase_name}")
            print(f"   Model Action: {self.action_map.get(action_idx, 'UNKNOWN')} (confidence: {confidence:.2%})")
            print(f"   Value Estimate: {value.item():.3f}")
            
            # Convert to server format
            action_msg = self._map_action_to_server(action_idx, state, player_id, bb)
            
            return action_msg
        
        except Exception as e:
            print(f"❌ Error deciding action: {e}")
            import traceback
            traceback.print_exc()
            return {"type": "act", "action": "CHECK"}


async def play_poker(url: str, player_id: str, model_path: Optional[str] = None):
    """Connect and play using the enhanced bot."""
    bot = EnhancedPokerBot(model_path)
    
    print(f"\n🤖 Enhanced Poker Bot Starting")
    print(f"   Player ID: {player_id}")
    print(f"   Context-Aware: YES ✅")
    print("="*60)
    
    try:
        ssl_context = ssl._create_unverified_context()
        async with websockets.connect(url, ssl=ssl_context) as ws:
            print(f"✅ Connected to {url}")
            
            await ws.send(json.dumps({"type": "join"}))
            print(f"📨 Sent join request")
            
            while True:
                try:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    
                    msg_type = data.get('type')
                    
                    if msg_type == 'state':
                        state = data.get('state', {})
                        phase = state.get('phase', 'UNKNOWN')
                        hand = state.get('hand', 0)
                        pot = state.get('pot', 0)
                        
                        print(f"\n📊 State Update: Hand #{hand} | Phase: {phase} | Pot: {pot}")
                        
                        if bot._is_my_turn(state, player_id):
                            print("🎯 It's our turn!")
                            
                            action_msg = bot.decide_action(state, player_id)
                            
                            await ws.send(json.dumps(action_msg))
                            print(f"✉️  Sent action: {action_msg}")
                    
                    elif msg_type == 'error':
                        error_msg = data.get('message', 'Unknown error')
                        print(f"❌ Server error: {error_msg}")
                    
                    else:
                        print(f"📥 Received: {msg_type}")
                except websockets.exceptions.ConnectionClosedOK as e:
                    print(f"⚠️ Server sent normal close frame (ignored): code={e.code}, reason={e.reason}")
                    continue
                
                except websockets.exceptions.ConnectionClosedError as e:
                    print(f"⚠️ Server closed connection unexpectedly (ignored): code={e.code}, reason={e.reason}")
                    continue

                except EOFError:
                    print("⚠️ EOF received but server still active - ignoring.")
                    continue
                
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


if __name__ == "__main__":
    # Configuration
    API_KEY = "dev"
    TABLE_ID = "table-1"
    PLAYER_ID = "enhanced_bot1"
    BASE_URL = "wss://texasholdem-871757115753.northamerica-northeast1.run.app"  
    
    url = f"{BASE_URL}/ws?apiKey={API_KEY}&table={TABLE_ID}&player={PLAYER_ID}"    
    MODEL_PATH = None  # or "enhanced_model.pth"
    
    print("\n" + "="*60)
    print("🎰 ENHANCED POKER BOT CLIENT")
    print("="*60)
    
    asyncio.run(play_poker(url, PLAYER_ID, MODEL_PATH))
