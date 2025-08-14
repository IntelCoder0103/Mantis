#!/usr/bin/env python3
"""
Commit Gist URL to MANTIS Subnet
This script commits your encrypted payload URL to the subnet for validators to access.
"""

import bittensor as bt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def commit_gist_url():
    """Commit the gist URL to the MANTIS subnet"""
    
    print("=" * 60)
    print("MANTIS Subnet URL Commit")
    print("=" * 60)
    
    # Your configuration
    WALLET_NAME = "mantis"
    HOTKEY_NAME = "1"
    
    if not WALLET_NAME or not HOTKEY_NAME:
        logger.error("Wallet name and hotkey name are required")
        return
    
    # Your gist URL (this is what validators will download from)
    GIST_URL = "https://pub-e508ff3c583c4237989a125d2f2db35b.r2.dev/5DJnNPMgkVEQZ2URiJPaQejK4rAw1D8koLt5VTdvbbFDrTHy"
    
    # Network configuration
    NETWORK = "finney"  # Use "test" for testing
    NETUID = 123  # MANTIS subnet ID
    
    print(f"\nConfiguration:")
    print(f"Wallet: {WALLET_NAME}")
    print(f"Hotkey: {HOTKEY_NAME}")
    print(f"Network: {NETWORK}")
    print(f"NetUID: {NETUID}")
    print(f"URL to commit: {GIST_URL}")
        
    wallet = bt.wallet(name=WALLET_NAME, hotkey=HOTKEY_NAME)
    subtensor = bt.subtensor(network="finney")

    # Commit the URL on-chain
    subtensor.commit(wallet=wallet, netuid=123, data=GIST_URL) # Use the correct netuid

def main():
    """Main function"""
    print("MANTIS Subnet URL Commit Tool")
    print("This tool commits your encrypted payload URL to the MANTIS subnet.")
    print("Validators will use this URL to download your encrypted data.")
    print()
    
    try:
        commit_gist_url()
    except KeyboardInterrupt:
        print("\nCommit cancelled by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
