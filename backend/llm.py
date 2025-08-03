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
        
        # Test with a simple prompt using gemini-1.5-flash
        model = genai.GenerativeModel('gemini-1.5-flash')
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
                "model": "gemini-1.5-flash"
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
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        result = response.text.strip() if hasattr(response, 'text') else str(response)
        logger.info(f"✅ Script generated successfully, length: {len(result)} characters")
        logger.debug(f"📜 Script preview: {result[:200]}...")
        
        return result
    except Exception as e:
        logger.error(f"❌ Failed to generate script: {str(e)}")
        raise Exception(f"Failed to generate script: {str(e)}") 

CONVERSATIONAL_SCRIPT_PROMPT = """
Write a 30-second hilarious comedy skit featuring Elon Musk and Donald Trump having an authentic heated discussion about this article.

CHARACTER PERSONALITIES:
- Trump: Confident, bombastic, uses phrases like "tremendous," "bigly," "believe me," "what the hell," "damn," "holy shit," often talks about deals, winning, and how things are "the best" or "the worst." Gets easily sidetracked into tangents about his achievements. Calls things "fucking incredible" when excited.
- Elon: Tech-focused, sometimes awkward, talks about Mars, Tesla, AI, makes references to memes, crypto, and "epic" things. Uses mild cuss words like "damn," "shit," "hell yeah," and "what the fuck" when explaining complex stuff. Often responds with technical explanations but in simple terms.

Follow this structure:
1. Trump opens with bombastic confusion about the article topic with mild cuss words
2. Elon tries to explain with tech analogies but gets frustrated and uses cuss words
3. Trump misunderstands completely and relates it to his achievements with profanity
4. Elon gets more animated and uses stronger language to explain
5. End with both getting heated but still trying to make their points

Tone:
- Authentic, heated but funny conversation with mild profanity
- Fast-paced, sharp, and character-driven
- Include their real speaking patterns and famous catchphrases
- Make them sound like they're having a real argument but keeping it entertaining

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
- Baburao: Classic Hera Pheri style. Always confused, speaks mix of Hindi-English. Uses phrases like "Ye kya bakchodi hai saala?", "Arre kameena", "Kya baat kar rahe ho saala", "Samajh nahi aa raha cutiyapa", "Kameene pareshaan kar diya", "Saala dimag kharab ho gaya". Gets everything completely wrong, makes absurd connections to daily life.
- Samay: Chess master comedian who speaks in Hindi-English mix. Uses words like "Yaar", "Bhai", "Dekho saala", "Samjha kameena", "Logic hai na saala", "Strategy chahiye cutiyapa". Patient but sarcastic, explains with chess analogies, uses modern slang and mild abuse.

DIALOGUE STYLE REQUIREMENTS:
- Write dialogue in ROMAN HINDI (Hinglish) - NOT pure English
- Use authentic Hindi words and phrases throughout
- Baburao style: "Arre yaar", "Ye kya bakchodi hai", "Samajh nahi aaya", "Dimag kharab kar diya"
- Samay style: "Dekho bhai", "Logic hai na", "Strategy chahiye", "Samjha", "Chess mein bolte hai na"
- Mix Hindi and English naturally like real Indian conversations
- Keep it authentic to how these characters actually speak

CONVERSATION STRUCTURE:
1. Baburao starts confused: "Ye kya bakchodi hai? Samajh nahi aa raha saala"
2. Samay explains in Hindi-English: "Arre saala, dekho simple hai"  
3. Baburao misunderstands completely with Hindi abuse: 
4. Samay uses chess analogy with mild cuss: "Bhai ye cutiyapa nahi hai, logic hai"
5. Baburao relates to daily problems: "Saala dimag kharab kar diya"
6. End with Baburao's classic confused abuse: "Kameene, samjha nahi aaya"

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
- Add mild Hindi cuss words like "saala," "kameena," "cutiyapa" where appropriate
- Make Baburao use classic abuse like "kameene," "saala," "saala kutta"
- Samay can use modern slang like "bhai ye kya cutiyapa hai," "saala samjha nahi"

EXAMPLE FORMAT:
Arre ye kya bakchodi hai saala? Samajh nahi aa raha yarr.

Dekho bhai, simple hai na. Logic lagao na chomu.

Saala dimag kharab kar diya. Kameene pareshaan kar diya.

Article: {article_text}

Generate ONLY clean, natural HINGLISH alternating dialogue with 4-6 short segments:
"""

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
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        result = response.text.strip() if hasattr(response, 'text') else str(response)
        logger.info(f"✅ {speaker_pair} conversational script generated successfully, length: {len(result)} characters")
        logger.debug(f"📜 Script preview: {result[:200]}...")
        
        return result
    except Exception as e:
        logger.error(f"❌ Failed to generate conversational script: {str(e)}")
        raise Exception(f"Failed to generate conversational script: {str(e)}")

TOPIC_SEARCH_PROMPT = """
You are a helpful assistant that finds or creates informative content about topics. Given a user's topic, you should:

1. FIRST try to find a real, recent article about this topic from reliable sources
2. If no recent article exists, create a comprehensive explanation about how the topic works

For topic: {topic}

Please provide:
1. A brief summary of what you found or created
2. The full content (either from the article or your generated explanation)
3. The source URL (if it's a real article) or "AI Generated" (if you created it)

Format your response as:
SUMMARY: [brief summary]
CONTENT: [full content]
SOURCE: [URL or "AI Generated"]

Focus on explaining HOW things work, their mechanisms, processes, and real-world applications.
"""

def search_or_generate_topic_content(topic: str) -> dict:
    """
    Search for real articles about a topic or generate comprehensive content about how it works.
    Returns a dictionary with summary, content, and source information.
    """
    try:
        logger.info(f"🔍 Searching/generating content for topic: {topic}")
        
        prompt = TOPIC_SEARCH_PROMPT.format(topic=topic)
        
        logger.info("🤖 Sending request to Gemini API for topic content...")
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        result = response.text.strip() if hasattr(response, 'text') else str(response)
        logger.info(f"✅ Topic content generated successfully, length: {len(result)} characters")
        
        # Parse the response to extract summary, content, and source
        lines = result.split('\n')
        summary = ""
        content = ""
        source = "AI Generated"
        
        current_section = None
        for line in lines:
            line = line.strip()
            if line.startswith("SUMMARY:"):
                current_section = "summary"
                summary = line.replace("SUMMARY:", "").strip()
            elif line.startswith("CONTENT:"):
                current_section = "content"
                content = line.replace("CONTENT:", "").strip()
            elif line.startswith("SOURCE:"):
                current_section = "source"
                source = line.replace("SOURCE:", "").strip()
            elif current_section == "content" and line:
                content += "\n" + line
            elif current_section == "summary" and line:
                summary += " " + line
        
        # If parsing failed, use the entire response as content
        if not content:
            content = result
            summary = f"Information about {topic}"
        
        logger.info(f"📄 Parsed content - Summary: {len(summary)} chars, Content: {len(content)} chars, Source: {source}")
        
        return {
            "summary": summary,
            "content": content,
            "source": source,
            "topic": topic
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to search/generate topic content: {str(e)}")
        raise Exception(f"Failed to search/generate topic content: {str(e)}")