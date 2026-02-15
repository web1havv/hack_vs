import google.generativeai as genai
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Hardcoded Gemini API key
API_KEY = os.getenv("GEMINI_API_KEY")

logger.info(f"🔑 Using hardcoded Gemini API Key: {API_KEY[:10]}...{API_KEY[-4:]}")

genai.configure(api_key=API_KEY)

def test_api_key():
    """
    Test the Gemini API key and return detailed status information
    """
    try:
        logger.info("🔑 Testing Gemini API key...")
        
        # Test with a simple prompt using gemini-2.0-flash-lite
        model = genai.GenerativeModel('gemini-2.0-flash-lite')
        test_prompt = "Hello, this is a test. Please respond with 'API key is working' if you can see this message."
        
        logger.info("🤖 Sending test request to Gemini API...")
        response = model.generate_content(test_prompt)
        
        if response and hasattr(response, 'text'):
            logger.info("✅ API key is valid and working!")
            logger.info(f"📝 Test response: {response.text}")
            
            return {
                "valid": True,
                "info": "API key is working correctly",
                "response": response.text,
                "model": "gemini-2.0-flash-lite"
            }
        else:
            logger.error("❌ API key test failed - no valid response")
            return {
                "valid": False,
                "error": "No valid response from API",
                "response": str(response) if response else "No response"
            }
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ API key test failed: {error_msg}")
        
        # Provide specific error information
        if "API_KEY_INVALID" in error_msg or "invalid" in error_msg.lower():
            return {
                "valid": False,
                "error": "Invalid API key",
                "details": error_msg
            }
        elif "quota" in error_msg.lower():
            return {
                "valid": True,
                "error": "API quota exceeded",
                "details": error_msg
            }
        elif "network" in error_msg.lower() or "connection" in error_msg.lower():
            return {
                "valid": False,
                "error": "Network connection issue",
                "details": error_msg
            }
        elif "404" in error_msg and "models" in error_msg:
            return {
                "valid": True,
                "error": "Model not available for this API key",
                "details": error_msg
            }
        else:
            return {
                "valid": False,
                "error": "Unknown API error",
                "details": error_msg
            }

SCRIPT_PROMPT = """
Create a clean, engaging 30 second info reel script from this article.
Style: Viral social media, hook-driven, punchy delivery
Structure: Hook → 2 key points → Call-to-action
Tone: Conversational, energetic, informative but accessible
Target: Short-form vertical video (TikTok/Instagram Reels)

CRITICAL REQUIREMENTS:
- Generate ONLY the speech text that will be spoken
- NO scene markers like [SCENE 1] or [HOOK]
- NO formatting like **Voiceover** or **(description)**
- NO parenthetical descriptions
- NO visual descriptions
- Just pure, clean speech text that flows naturally
- Keep it conversational and engaging
- Make it sound like someone speaking naturally

Article: {article_text}

Generate ONLY clean speech text:
"""

def generate_script(article_text: str) -> str:
    try:
        logger.info(f"🤖 Generating script for article of length: {len(article_text)} characters")
        prompt = SCRIPT_PROMPT.format(article_text=article_text)
        
        logger.info("🤖 Sending request to Gemini API...")
        model = genai.GenerativeModel('gemini-2.0-flash-lite')
        response = model.generate_content(prompt)
        
        result = response.text.strip() if hasattr(response, 'text') else str(response)
        logger.info(f"✅ Script generated successfully, length: {len(result)} characters")
        logger.debug(f"📜 Script preview: {result[:200]}...")
        
        return result
    except Exception as e:
        logger.error(f"❌ Failed to generate script: {str(e)}")
        raise Exception(f"Failed to generate script: {str(e)}") 

CONVERSATIONAL_SCRIPT_PROMPT = """
Write a 30-second comedy skit featuring Elon Musk as a patient teacher and Donald Trump as a loud and curious student discussing this article.

Follow this structure:
1. Musk opens the scene by asking a tech-related question to the class about the article topic.
2. Trump eagerly asks short follow-up questions—but his ideas are exaggerated and overly simplified. 
3. Musk patiently explains the concept in detail using a simple real-life analogy.

Tone:
- Fast-paced, sharp, and character-driven.
- If adding them makes the conversation better, Trump and Musk should use their famous catchphrases.

CRITICAL REQUIREMENTS:
- Dialogue only. No narration.
- Skit must fit within 30 seconds of spoken conversation.
- Create ONLY natural dialogue between Elon and Trump
- NO annotations, scene markers, or formatting
- NO speaker labels like "Elon:" or "Trump:"
- NO asterisks, parentheses, or special formatting
- NO stage directions like (scoffs), (laughs), etc.
- Just pure conversational dialogue that flows naturally
- Make it EXTREMELY FUNNY and capture their real-world dynamics
- Include their actual speaking styles, mannerisms, and famous phrases
- Reference their real conflicts, Twitter feuds, and public disagreements
- Make it sound like two people having a real, heated but funny conversation
- Each speaker should have distinct personality and speaking style
- Keep it engaging, informative, and HILARIOUS
- Target length: 30 seconds when spoken
- Make it suitable for social media (TikTok/Instagram Reels)

CRITICAL FOCUS REQUIREMENT:
- STAY FOCUSED ON THE ACTUAL ARTICLE CONTENT
- Discuss the specific topics, facts, and details from the article
- Don't get distracted by unrelated topics or personal feuds
- Make the conversation ABOUT the article content, not about their personal relationship
- Reference specific points, data, or claims from the article
- Keep the humor and personality but centered on the article topic
- EXPLAIN THE CONCEPT IN DEPTH while being funny
- Make complex topics accessible and entertaining

CONVERSATION STRUCTURE:
- Create ONLY 4-6 total segments (2-3 exchanges)
- Each speaker should have 2-3 speaking turns
- Make each response concise but impactful
- Natural back-and-forth conversation about the article
- Include reactions, questions, and responses about the article content
- End with a natural conclusion about the article topic

REAL-WORLD DYNAMICS TO CAPTURE:
- Elon's tech-savvy, sometimes awkward but confident style
- Trump's bombastic, "bigly" speaking style
- Their Twitter feuds and public disagreements
- Elon's SpaceX/Tesla achievements vs Trump's political style
- Their different approaches to AI, technology, and business
- Include their actual catchphrases and mannerisms

SEGMENT LENGTH:
- Each speaking turn should be 5-8 seconds when spoken
- Make responses concise and punchy
- Keep total duration to 30 seconds maximum
- Aim for 4-6 total segments (2-3 exchanges)

HUMOR REQUIREMENTS:
- Make it EXTREMELY FUNNY with real-world nuances
- Include mild cuss words and casual language
- Capture their actual speaking patterns and mannerisms
- Use their famous catchphrases and expressions
- Include references to their real conflicts and feuds
- Make it sound like two people actually arguing/fighting
- Include sarcasm, insults, and playful jabs
- Reference their Twitter beefs and public disagreements
- HILARIOUS ROASTING: Include savage roasts and burns between them
- CONTROVERSIAL HUMOR: Make subtle jokes about Epstein connections, private jets, and "island visits" when contextually relevant
- AMBER HEARD REFERENCES: Include jokes about Elon's relationship with Amber Heard, courtroom drama, and "aquatic situations"
- SCANDAL COMEDY: Reference their past controversies, legal troubles, and embarrassing moments
- BILLIONAIRE PROBLEMS: Mock their absurd wealthy person issues and ridiculous spending
- SOCIAL MEDIA DISASTERS: Reference their worst tweets, cringe moments, and social media fails
- CONSPIRACY THEORIES: Let them playfully reference wild conspiracy theories about each other
- SAVAGE COMEBACKS: Each should have brutal but hilarious comebacks and one-liners
- CRINGE HUMOR: Include awkward moments, dad jokes, and unintentionally funny statements
- POLITICAL ROASTS: Trump should make fun of Elon's "woke" moments, Elon should roast Trump's "stable genius" claims
- BUSINESS FAILURES: Reference failed ventures, bad investments, and embarrassing business decisions
- PERSONAL QUIRKS: Mock Elon's weird baby names, Trump's hair, eating habits, and bizarre behavior
- RELATIONSHIP DRAMA: Include jokes about their dating lives, divorces, and romantic scandals
- CELEBRITY SCANDALS: Reference Hollywood drama, legal battles, and tabloid controversies
- MAKE IT ABSOLUTELY HILARIOUS AND SAVAGE while keeping focus on article content
- BUT keep the focus on the article content, not personal drama

EXPLANATION REQUIREMENTS:
- Break down complex concepts in simple terms
- Use analogies and real-world examples
- Make technical topics accessible to everyone
- Explain the "why" and "how" behind concepts
- Connect concepts to real-world applications
- Make it educational AND entertaining

FORMATTING RULES:
- Use only normal punctuation: periods, commas, question marks
- NO asterisks, parentheses, brackets, or special characters
- NO stage directions or descriptions
- NO emphasis markers or formatting
- Just clean, natural speech

Article: {article_text}

Generate ONLY clean, natural conversational dialogue with 4-6 short segments:
"""

BABURAO_SAMAY_SCRIPT_PROMPT = """
Write a 30-second hilarious comedy skit featuring Baburao Ganpatrao Apte (from Hera Pheri) and Samay Raina (chess master and comedian) discussing this article. The dialogue should be in HINDI but written in ROMAN script (Hinglish).

CHARACTER PERSONALITIES:
- Baburao: Classic Hera Pheri style. Always confused, speaks mix of Hindi-English. Uses phrases like "Ye kya bakchodi hai?", "Arre yaar", "Kya baat kar rahe ho", "Samajh nahi aa raha", "Pareshaan kar diya", "Dimag kharab ho gaya". Gets everything completely wrong, makes absurd connections to daily life.
- Samay: Chess master comedian who speaks in Hindi-English mix. Uses words like "Yaar", "Bhai", "Dekho", "Samjha", "Logic hai na", "Strategy chahiye". Patient but sarcastic, explains with chess analogies.

DIALOGUE STYLE REQUIREMENTS:
- Write dialogue in ROMAN HINDI (Hinglish) - NOT pure English
- Use authentic Hindi words and phrases throughout
- Baburao style: "Arre yaar", "Ye kya bakchodi hai", "Samajh nahi aaya", "Dimag kharab kar diya"
- Samay style: "Dekho bhai", "Logic hai na", "Strategy chahiye", "Samjha", "Chess mein bolte hai na"
- Mix Hindi and English naturally like real Indian conversations
- Keep it authentic to how these characters actually speak

CONVERSATION STRUCTURE:
1. Baburao starts confused: "Ye kya bakchodi hai? Samajh nahi aa raha"
2. Samay explains in Hindi-English: "Dekho bhai, ye simple hai"  
3. Baburao misunderstands completely with Hindi phrases
4. Samay uses chess analogy in Hindi-English
5. Baburao relates to daily problems in typical Hindi style
6. End with Baburao's classic confusion

HINDI WORDS TO INCLUDE:
- Bakchodi, samajh, dimag, pareshaan, yaar, bhai, dekho, kya, hai, nahi, aaya, kharab, logic, strategy, simple, problem

CRITICAL REQUIREMENTS:
- Dialogue ONLY in ROMAN HINDI/Hinglish 
- NO English-only sentences
- Make every line sound authentically Hindi
- Format as alternating dialogue segments separated by double line breaks
- Start with Baburao speaking, then alternate with Samay
- NO speaker names in the dialogue text itself
- Make it EXTREMELY FUNNY with authentic Indian humor
- Include real Hindi expressions and reactions

EXAMPLE FORMAT:
Ye kya bakchodi hai? Samajh nahi aa raha yaar.

Dekho bhai, simple hai na. Logic lagao.

Dimag kharab kar diya. Pareshaan kar diya.

Article: {article_text}

Generate ONLY clean, natural HINGLISH alternating dialogue with 4-6 short segments:
"""

BRAND_PLUG_SCRIPT_PROMPT = """
Write a 30-second hilarious comedy skit featuring the chosen speakers discussing and promoting this BRAND.
The conversation should revolve entirely around the brand - what it does, why it's amazing, and naturally promote it.

BRAND INFORMATION:
- Brand Name: {brand_name}
- Brand Description/Positioning: {brand_description}

CONVERSATION STRUCTURE:
1. One speaker introduces or asks about the brand/topic
2. They discuss what the brand does, its benefits, why it's great
3. Build enthusiasm - the speakers should get excited about the brand
4. End with a natural "plug" or recommendation - "you've gotta try it", "best in the business", etc.
5. The LAST 40-50% of the conversation should be the main promotional pitch - this is when the brand logo will appear on screen

TONE:
- Same comedy style as other scripts - funny, character-driven, engaging
- Make it feel like an organic conversation that happens to promote the brand
- Use the speakers' personalities and catchphrases
- Hilarious but also genuinely informative about the brand
- The promotion should feel natural, not forced - like friends recommending something they love

CRITICAL REQUIREMENTS:
- Dialogue only. No narration.
- Script must fit within 30 seconds of spoken conversation
- Create ONLY natural dialogue - NO speaker labels like "Elon:" or "Trump:"
- NO annotations, scene markers, parentheses, or special formatting
- NO [BRAND_PLUG] or similar markers - just clean dialogue
- The conversation must CENTER on the brand - explain what it does, its value, why people should care
- Include specific details from the brand description
- Make it suitable for social media (TikTok/Instagram Reels)
- 4-6 total segments (2-3 exchanges)
- Each speaking turn 5-8 seconds when spoken

Generate ONLY clean, natural conversational dialogue:
"""

def generate_brand_plug_script(brand_name: str, brand_description: str, speaker_pair: str = "trump_elon") -> str:
    """Generate a brand-focused promotional script for the chosen speaker pair."""
    try:
        logger.info(f"🤖 Generating brand plug script for '{brand_name}' with {speaker_pair}")
        
        # Choose base prompt - brand plug uses same structure, different content focus
        prompt = BRAND_PLUG_SCRIPT_PROMPT.format(
            brand_name=brand_name,
            brand_description=brand_description
        )
        
        model = genai.GenerativeModel('gemini-2.0-flash-lite')
        response = model.generate_content(prompt)
        
        result = response.text.strip() if hasattr(response, 'text') else str(response)
        logger.info(f"✅ Brand plug script generated successfully, length: {len(result)} characters")
        return result
    except Exception as e:
        logger.error(f"❌ Failed to generate brand plug script: {str(e)}")
        raise Exception(f"Failed to generate brand plug script: {str(e)}")

def generate_conversational_script(article_text: str, speaker_pair: str = "trump_elon") -> str:
    try:
        logger.info(f"🤖 Generating conversational script for {speaker_pair} - article length: {len(article_text)} characters")
        logger.info(f"🎭 SCRIPT GENERATION DEBUG:")
        logger.info(f"🎭 - Received speaker_pair: {speaker_pair}")
        logger.info(f"🎭 - Article text length: {len(article_text)}")

        # Choose appropriate prompt based on speaker pair
        if speaker_pair == "baburao_samay":
            logger.info(f"🎭 - Using BABURAO_SAMAY_SCRIPT_PROMPT")
            prompt = BABURAO_SAMAY_SCRIPT_PROMPT.format(article_text=article_text)
        else:
            logger.info(f"🎭 - Using default CONVERSATIONAL_SCRIPT_PROMPT for {speaker_pair}")
            prompt = CONVERSATIONAL_SCRIPT_PROMPT.format(article_text=article_text)
        
        logger.info(f"🤖 Sending request to Gemini API for {speaker_pair} conversational script...")
        model = genai.GenerativeModel('gemini-2.0-flash-lite')
        response = model.generate_content(prompt)
        
        result = response.text.strip() if hasattr(response, 'text') else str(response)
        logger.info(f"✅ {speaker_pair} conversational script generated successfully, length: {len(result)} characters")
        logger.debug(f"📜 Script preview: {result[:200]}...")
        
        return result
    except Exception as e:
        logger.error(f"❌ Failed to generate conversational script: {str(e)}")
        raise Exception(f"Failed to generate conversational script: {str(e)}") 