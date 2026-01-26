import os
import json
import argparse
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path so we can find src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orkg_client import ORKGClient
from src.template_mapper import TemplateMapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("append_tool")

def main():
    parser = argparse.ArgumentParser(description="Append extracted data to a SPECIFIC ORKG Paper ID")
    parser.add_argument("--file", required=True, help="Path to the JSON file")
    parser.add_argument("--paper-id", required=True, help="Target ORKG Paper ID (e.g., R1568688)")
    parser.add_argument("--host", default="sandbox", help="ORKG host (default: sandbox)")
    
    args = parser.parse_args()
    
    # 1. Load credentials
    load_dotenv()
    email = os.getenv("ORKG_EMAIL")
    password = os.getenv("ORKG_PASSWORD")
    
    if not email or not password:
        logger.error("Missing ORKG credentials.")
        return

    # 2. Initialize components
    logger.info(f"Initializing ORKG client for {args.host}...")
    orkg_client = ORKGClient(host=args.host, email=email, password=password)
    
    # 3. Load data
    try:
        with open(args.file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded data from {args.file}")
    except Exception as e:
        logger.error(f"Failed to load JSON file: {e}")
        return

    # 4. Extract contribution data (assuming mapped_to_orkg exists)
    if "mapped_to_orkg" not in data or "contributions" not in data["mapped_to_orkg"]:
        logger.error("JSON does not contain 'mapped_to_orkg.contributions'. Please check file format.")
        # Try fallbacks?
        return

    contributions = data["mapped_to_orkg"]["contributions"]
    logger.info(f"Found {len(contributions)} contributions to append.")
    
    # 5. Append each contribution to the target paper
    for contrib in contributions:
        label = contrib.get("label", "Unknown Model")
        logger.info(f"Attempting to append contribution: '{label}' to Paper {args.paper_id}")
        
        # We call the client method directly to use the merge_if_exists logic
        # ORKGClient.add_contribution_to_paper(paper_id, contribution_data)
        
        new_cid = orkg_client.add_contribution_to_paper(args.paper_id, contrib)
        
        if new_cid:
            logger.info(f"✅ SUCCESS! Added contribution '{label}' with ID: {new_cid}")
            print(f"\n✅ Contribution '{label}' successfully added/merged.")
            print(f"🔗 View Paper: https://sandbox.orkg.org/paper/{args.paper_id}")
            print(f"🔗 View Contribution: https://sandbox.orkg.org/contribution/{new_cid}")
        else:
            logger.error(f"[X] FAILED to add contribution '{label}'")

if __name__ == "__main__":
    main()
