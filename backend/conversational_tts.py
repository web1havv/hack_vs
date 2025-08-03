import numpy as np
import soundfile as sf
import os
import tempfile
import logging
import re
import requests
from datetime import datetime
import json

logger = logging.getLogger(__name__)

# Speaker Configuration with Multiple API Keys and Voice IDs
SPEAKER_CONFIG = {
    "trump": {
        "voice_id": "ANHNqAseXGR3gBQps4vo",  # Trump voice from ElevenLabs
        "api_key_env": "ELEVENLABS_API_KEY_TRUMP",  # Separate API key for Trump
        "fallback_api_key_env": "ELEVENLABS_API_KEY"  # Fallback to main key
    },
    "elon": {
        "voice_id": "uTsZKELdeD1KQZz4M45o",   # Elon Musk voice from ElevenLabs (updated)
        "api_key_env": "ELEVENLABS_API_KEY_ELON",   # Separate API key for Elon
        "fallback_api_key_env": "ELEVENLABS_API_KEY"  # Fallback to main key
    },
    "samay": {
        "voice_id": "QhMdw7peUi09bf5eDE34",  # Samay Raina voice ID
        "api_key_env": "ELEVENLABS_API_KEY_BABURAO_SAMAY",  # New API key for Baburao & Samay
        "fallback_api_key_env": "ELEVENLABS_API_KEY"  # Fallback to main key
    },
    "baburao": {
        "voice_id": "o76izsJbtLZKHDxpMquz",  # Baburao voice ID
        "api_key_env": "ELEVENLABS_API_KEY_BABURAO_SAMAY",  # New API key for Baburao & Samay
        "fallback_api_key_env": "ELEVENLABS_API_KEY"  # Fallback to main key
    }
}

# Speaker Pair Configurations for Frontend
SPEAKER_PAIRS = {
    "trump_elon": {
        "name": "Trump & Elon",
        "speakers": ["trump", "elon"],
        "description": "Political discussions with tech innovation"
    },
    "baburao_samay": {
        "name": "Baburao & Samay",
        "speakers": ["baburao", "samay"], 
        "description": "Comedy legend meets chess master insights"
    }
}

# Legacy voice IDs for backward compatibility
TRUMP_VOICE_ID = SPEAKER_CONFIG["trump"]["voice_id"]
ELON_VOICE_ID = SPEAKER_CONFIG["elon"]["voice_id"]

def parse_conversational_script(script_text, speaker_pair="trump_elon"):
    """
    Parse a conversational script and separate it into speakers
    Supports both explicit speaker format (**Speaker:** text) and auto-alternating
    
    Args:
        script_text: The text to parse
        speaker_pair: Key from SPEAKER_PAIRS (e.g., "trump_elon", "modi_elon", "tanmay_samay")
    
    Returns a list of tuples: [(speaker, text), (speaker, text), ...]
    """
    try:
        logger.info("🔍 Parsing conversational script for speaker separation")
        logger.info(f"📝 Script length: {len(script_text)} characters")
        
        # Check if script has explicit speaker markers (e.g., **Trump:** format)
        speaker_pattern = r'\*\*([^*]+):\*\*\s*([^*]+?)(?=\*\*[^*]+:\*\*|\Z)'
        explicit_speakers = re.findall(speaker_pattern, script_text, re.DOTALL)
        
        if explicit_speakers:
            logger.info("📢 Found explicit speaker markers in script")
            speakers = []
            for speaker_name, text in explicit_speakers:
                # Normalize speaker names to lowercase for consistency
                speaker_name = speaker_name.lower().strip()
                text = text.strip()
                
                if text:  # Only add non-empty text
                    speakers.append((speaker_name, text))
                    logger.info(f"   📝 {speaker_name}: {len(text)} characters")
            
            logger.info(f"✅ Parsed script into {len(speakers)} explicit speaker segments")
            return speakers
        
        # Fallback to auto-alternating system for scripts without explicit speakers
        logger.info("🔄 No explicit speakers found, using auto-alternating system")
        
        # Split by double line breaks first (for paragraph-style segments)
        paragraphs = [p.strip() for p in script_text.split('\n\n') if p.strip()]
        
        if paragraphs and len(paragraphs) >= 2:
            # Use paragraph-based splitting for better segment control
            logger.info(f"📝 Found {len(paragraphs)} dialogue paragraphs")
            segments = paragraphs
        else:
            # Fallback: split by sentences and alternate speakers  
            segments = re.split(r'[.!?]+', script_text)
            segments = [s.strip() for s in segments if s.strip()]
            logger.info(f"📝 Using sentence-based splitting: {len(segments)} segments")
        
        # Combine into longer segments if needed (6-8 total segments)
        target_segments = 6  # Aim for 6 segments total
        if len(segments) > target_segments * 2:
            # Too many segments, combine them
            segments_per_chunk = max(1, len(segments) // target_segments)
        else:
            segments_per_chunk = 1
        
        # Alternate between speakers from selected pair
        speakers = []
        if speaker_pair in SPEAKER_PAIRS:
            pair_speakers = SPEAKER_PAIRS[speaker_pair]["speakers"]
        else:
            # Fallback to trump-elon
            pair_speakers = ["trump", "elon"]
            logger.warning(f"Unknown speaker pair '{speaker_pair}', using trump-elon")
        
        current_speaker = pair_speakers[0]
        current_segment = ""
        segment_count = 0
        
        for i, segment in enumerate(segments):
            if segment:
                current_segment += segment + " "
                
                # Create a segment when we have enough content or reach the end
                if (len(current_segment.strip()) > 50 and 
                    (i + 1) % segments_per_chunk == 0) or i == len(segments) - 1:
                    
                    if current_segment.strip():
                        speakers.append((current_speaker, current_segment.strip()))
                        segment_count += 1
                        # Switch to next speaker in pair
                        current_index = pair_speakers.index(current_speaker)
                        next_index = (current_index + 1) % len(pair_speakers)
                        current_speaker = pair_speakers[next_index]
                        current_segment = ""
        
        # If we still have too many segments, combine some
        if len(speakers) > 8:
            combined_speakers = []
            for i in range(0, len(speakers), 2):
                if i + 1 < len(speakers):
                    # Combine two segments from the same speaker
                    combined_text = speakers[i][1] + " " + speakers[i+1][1]
                    combined_speakers.append((speakers[i][0], combined_text))
                else:
                    combined_speakers.append(speakers[i])
            speakers = combined_speakers
        
        logger.info(f"✅ Parsed script into {len(speakers)} speaker segments")
        logger.debug(f"📝 Speaker breakdown: {speakers[:3]}...")  # Show first 3 segments
        
        return speakers
    except Exception as e:
        logger.error(f"❌ Failed to parse conversational script: {str(e)}")
        raise Exception(f"Failed to parse conversational script: {str(e)}")

def get_api_key_for_speaker(speaker_name):
    """Get the appropriate API key for a specific speaker"""
    try:
        # Special handling for Baburao & Samay with new API key
        if speaker_name in ["baburao", "samay"]:
            # Use the new API key for Baburao & Samay
            new_api_key = "sk_f800d01dcf584b4a637e0363c12430d7c5f32a6289b0785f"
            logger.info(f"🔑 Using new ElevenLabs API key for {speaker_name}")
            return new_api_key
        
        if speaker_name in SPEAKER_CONFIG:
            config = SPEAKER_CONFIG[speaker_name]
            # Try speaker-specific API key first
            api_key = os.getenv(config["api_key_env"], "")
            if api_key:
                logger.info(f"🔑 Using speaker-specific API key for {speaker_name}")
                return api_key
            
            # Fallback to main API key
            fallback_key = os.getenv(config["fallback_api_key_env"], "")
            if fallback_key:
                logger.info(f"🔑 Using fallback API key for {speaker_name}")
                return fallback_key
        
        # Default fallback
        default_key = os.getenv("ELEVENLABS_API_KEY", "")
        if default_key:
            logger.info(f"🔑 Using default API key for {speaker_name}")
            return default_key
        
        logger.warning(f"⚠️ No API key found for speaker {speaker_name}")
        return ""
        
    except Exception as e:
        logger.error(f"❌ Error getting API key for {speaker_name}: {str(e)}")
        return os.getenv("ELEVENLABS_API_KEY", "")

def generate_elevenlabs_voice_segment(text, voice_id, output_path, speaker_name=None):
    """Generate voice segment using ElevenLabs API with speaker-specific API key"""
    try:
        logger.info(f"🎤 Generating voice segment for speaker: {speaker_name}")
        logger.info(f"🆔 Voice ID: {voice_id}")
        logger.info(f"📝 Text: {text[:50]}...")
        
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        
        # Get appropriate API key for this speaker
        api_key = get_api_key_for_speaker(speaker_name) if speaker_name else os.getenv("ELEVENLABS_API_KEY", "")
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key
        }
        
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }
        
        logger.info("🌐 Sending request to ElevenLabs API...")
        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            file_size = os.path.getsize(output_path)
            logger.info(f"✅ Voice segment generated successfully")
            logger.info(f"📊 File size: {file_size} bytes")
            return True
        else:
            logger.warning(f"⚠️ ElevenLabs failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"❌ ElevenLabs error: {str(e)}")
        return False

def batch_generate_elevenlabs_voice_segments(segments_data, output_dir):
    """
    Batch generate multiple voice segments using ElevenLabs API with better error handling
    segments_data: list of (text, voice_id, output_filename, speaker_name) tuples
    """
    try:
        logger.info(f"🎤 Batch generating {len(segments_data)} voice segments")
        
        # Process in smaller batches to avoid rate limits
        batch_size = 3  # Process 3 segments at a time
        successful_segments = []
        
        for i in range(0, len(segments_data), batch_size):
            batch = segments_data[i:i + batch_size]
            logger.info(f"🎤 Processing batch {i//batch_size + 1}/{(len(segments_data) + batch_size - 1)//batch_size}")
            
            for j, segment_data in enumerate(batch):
                # Handle both old and new tuple formats for backward compatibility
                if len(segment_data) == 4:
                    text, voice_id, filename, speaker_name = segment_data
                else:
                    text, voice_id, filename = segment_data
                    speaker_name = None
                
                segment_index = i + j
                output_path = os.path.join(output_dir, filename)
                
                logger.info(f"🎤 Generating segment {segment_index + 1}/{len(segments_data)} for speaker: {speaker_name}")
                logger.info(f"🆔 Voice ID: {voice_id}")
                logger.info(f"📝 Text: {text[:50]}...")
                
                # Make API request with speaker-specific API key
                url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
                
                # Get appropriate API key for this speaker
                api_key = get_api_key_for_speaker(speaker_name) if speaker_name else os.getenv("ELEVENLABS_API_KEY", "")
                
                headers = {
                    "Accept": "audio/mpeg",
                    "Content-Type": "application/json",
                    "xi-api-key": api_key
                }
                
                data = {
                    "text": text,
                    "model_id": "eleven_monolingual_v1",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75
                    }
                }
                
                try:
                    logger.info("🌐 Sending request to ElevenLabs API...")
                    response = requests.post(url, json=data, headers=headers, timeout=30)
                    
                    if response.status_code == 200:
                        with open(output_path, 'wb') as f:
                            f.write(response.content)
                        
                        file_size = os.path.getsize(output_path)
                        logger.info(f"✅ Voice segment generated successfully")
                        logger.info(f"📊 File size: {file_size} bytes")
                        
                        # Extract speaker from filename
                        speaker = "elon" if "elon" in filename else "trump"
                        successful_segments.append((speaker, output_path))
                        
                    elif response.status_code == 401:
                        error_data = response.json()
                        if "quota_exceeded" in str(error_data):
                            logger.error(f"❌ ElevenLabs quota exceeded: {error_data}")
                            logger.error("💡 Consider upgrading your ElevenLabs plan or using a different API key")
                            break
                        else:
                            logger.warning(f"⚠️ ElevenLabs authentication failed: {response.status_code} - {response.text}")
                            continue
                            
                    elif response.status_code == 429:
                        logger.warning(f"⚠️ Rate limit hit, waiting 2 seconds...")
                        import time
                        time.sleep(2)
                        continue
                        
                    else:
                        logger.warning(f"⚠️ ElevenLabs failed: {response.status_code} - {response.text}")
                        continue
                        
                except requests.exceptions.Timeout:
                    logger.warning(f"⚠️ Request timeout for segment {segment_index + 1}, retrying...")
                    import time
                    time.sleep(1)
                    continue
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to generate segment {segment_index + 1}: {str(e)}")
                    continue
                
                # Small delay between requests to be respectful
                import time
                time.sleep(0.5)
            
            # Delay between batches
            if i + batch_size < len(segments_data):
                logger.info("⏳ Waiting 1 second between batches...")
                import time
                time.sleep(1)
        
        logger.info(f"✅ Successfully generated {len(successful_segments)} out of {len(segments_data)} segments")
        return successful_segments
        
    except Exception as e:
        logger.error(f"❌ Failed to batch generate voice segments: {str(e)}")
        return []

def combine_audio_segments(segments, output_path):
    """
    Combine multiple audio segments into one file using ffmpeg
    segments: list of (speaker, audio_path) tuples
    """
    try:
        logger.info(f"🔗 Combining {len(segments)} audio segments")
        
        # Create a temporary file list for ffmpeg
        file_list_path = tempfile.mktemp(suffix=".txt")
        
        with open(file_list_path, 'w') as f:
            for speaker, audio_path in segments:
                if os.path.exists(audio_path):
                    # Escape the path for ffmpeg
                    escaped_path = audio_path.replace("'", "'\\''")
                    f.write(f"file '{escaped_path}'\n")
        
        # Use ffmpeg to concatenate all audio files
        import subprocess
        
        cmd = [
            './ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', file_list_path,
            '-c', 'copy',
            output_path,
            '-y'  # Overwrite output file
        ]
        
        logger.info(f"🎵 Running ffmpeg command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            file_size = os.path.getsize(output_path)
            logger.info(f"✅ Audio segments combined successfully using ffmpeg")
            logger.info(f"📊 Combined file size: {file_size} bytes")
            
            # Clean up temporary file list
            try:
                os.remove(file_list_path)
            except:
                pass
                
            return True
        else:
            logger.error(f"❌ ffmpeg failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to combine audio segments: {str(e)}")
        return False

def generate_conversational_voiceover(script_text, output_path=None, speaker_pair="trump_elon"):
    """
    Generate conversational audio with alternating speakers (optimized version)
    
    Args:
        script_text: The script to convert to audio
        output_path: Where to save the audio file
        speaker_pair: Which speaker pair to use (trump_elon, modi_elon, tanmay_samay)
    """
    try:
        request_id = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"🎵 [{request_id}] Starting optimized conversational voiceover generation")
        logger.info(f"📝 [{request_id}] Script length: {len(script_text)} characters")
        
        if not output_path:
            output_path = tempfile.mktemp(suffix=".wav")
            logger.info(f"📁 [{request_id}] Using temporary output path: {output_path}")
        
        # Parse script into speaker segments with selected pair
        logger.info(f"🔍 [{request_id}] Parsing script for speaker separation...")
        logger.info(f"🎭 [{request_id}] Using speaker pair: {speaker_pair}")
        speaker_segments = parse_conversational_script(script_text, speaker_pair)
        
        # Prepare batch data for ElevenLabs
        logger.info(f"🎤 [{request_id}] Preparing batch requests for ElevenLabs...")
        batch_data = []
        
        for i, (speaker, text) in enumerate(speaker_segments):
            # Get voice ID from speaker configuration
            if speaker in SPEAKER_CONFIG:
                voice_id = SPEAKER_CONFIG[speaker]["voice_id"]
            else:
                # Fallback to legacy system for unknown speakers
                voice_id = ELON_VOICE_ID if speaker == "elon" else TRUMP_VOICE_ID
                logger.warning(f"⚠️ [{request_id}] Unknown speaker '{speaker}', using fallback voice ID")
            
            filename = f"segment_{i+1}_{speaker}.wav"
            # Include speaker name in the batch data for API key selection
            batch_data.append((text, voice_id, filename, speaker))
        
        # Create temporary directory for segments
        temp_dir = tempfile.mkdtemp()
        logger.info(f"📁 [{request_id}] Using temporary directory: {temp_dir}")
        
        # Batch generate all voice segments
        audio_segments = batch_generate_elevenlabs_voice_segments(batch_data, temp_dir)
        
        if not audio_segments:
            logger.error(f"❌ [{request_id}] No audio segments were generated successfully")
            raise Exception("No audio segments were generated")
        
        # Combine all segments
        logger.info(f"🔗 [{request_id}] Combining {len(audio_segments)} audio segments...")
        if combine_audio_segments(audio_segments, output_path):
            logger.info(f"✅ [{request_id}] Conversational voiceover generated successfully")
            
            # Clean up temporary segment files
            for speaker, segment_path in audio_segments:
                try:
                    os.remove(segment_path)
                    logger.debug(f"🗑️ [{request_id}] Removed temporary segment: {segment_path}")
                except Exception as e:
                    logger.warning(f"⚠️ [{request_id}] Failed to remove temporary file {segment_path}: {str(e)}")
            
            # Clean up temporary directory
            try:
                import shutil
                shutil.rmtree(temp_dir)
                logger.debug(f"🗑️ [{request_id}] Removed temporary directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"⚠️ [{request_id}] Failed to remove temporary directory {temp_dir}: {str(e)}")
            
            return output_path
        else:
            raise Exception("Failed to combine audio segments")
            
    except Exception as e:
        logger.error(f"❌ [{request_id}] Failed to generate conversational voiceover: {str(e)}")
        logger.error(f"❌ [{request_id}] Error type: {type(e).__name__}")
        raise Exception(f"Failed to generate conversational voiceover: {str(e)}")

def create_speaker_timeline(script_text, speaker_pair="trump_elon"):
    """
    Create a timeline of when each speaker is talking
    Returns list of (speaker, start_time, end_time, text) tuples
    """
    try:
        logger.info("⏰ Creating speaker timeline")
        logger.info(f"🎭 TIMELINE CREATION - speaker_pair: {speaker_pair}")
        
        # Parse script into speaker segments with speaker_pair
        speaker_segments = parse_conversational_script(script_text, speaker_pair)
        logger.info(f"🎭 TIMELINE - speaker_segments: {speaker_segments}")
        
        timeline = []
        current_time = 0.0
        
        for speaker, text in speaker_segments:
            # Estimate duration (roughly 0.5 seconds per word)
            word_count = len(text.split())
            duration = word_count * 0.5
            
            timeline.append({
                'speaker': speaker,
                'start_time': current_time,
                'end_time': current_time + duration,
                'text': text
            })
            
            current_time += duration + 0.2  # Add 0.2s pause between speakers
        
        logger.info(f"✅ Created timeline with {len(timeline)} segments")
        return timeline
        
    except Exception as e:
        logger.error(f"❌ Failed to create speaker timeline: {str(e)}")
        raise Exception(f"Failed to create speaker timeline: {str(e)}") 