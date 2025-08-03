"""
Caption Processing Module for Video Generation
Phase 1: Enhanced Timeline with Word-Level Timing and Caption Chunks
"""

import logging
import re
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class CaptionProcessor:
    """
    Processes conversational timelines to generate caption data
    """
    
    def __init__(self):
        # Caption display settings
        self.max_chars_per_line = 50  # Characters per caption line
        self.max_duration_per_caption = 4.0  # Max seconds per caption
        self.min_duration_per_caption = 1.0  # Min seconds per caption
        self.words_per_second = 2.0  # Average speaking rate (words/second)
        
        logger.info("🎬 Caption processor initialized")
    
    def estimate_word_timing(self, text: str, start_time: float, end_time: float) -> List[Dict]:
        """
        Estimate individual word timing within a text segment
        
        Args:
            text: The text segment  
            start_time: Segment start time in seconds
            end_time: Segment end time in seconds
            
        Returns:
            List of word timing dictionaries: [{"word": str, "start": float, "end": float}, ...]
        """
        try:
            # Clean and split text into words
            words = re.findall(r'\S+', text)  # Split on whitespace, keep punctuation
            if not words:
                return []
            
            total_duration = end_time - start_time
            words_per_second = len(words) / total_duration if total_duration > 0 else 2.0
            
            word_timings = []
            current_time = start_time
            
            for word in words:
                # Estimate word duration based on character length and speaking rate
                char_count = len(word)
                base_duration = 1.0 / words_per_second
                
                # Adjust for word length (longer words take slightly more time)
                duration_adjustment = (char_count - 4) * 0.05 if char_count > 4 else 0
                word_duration = max(0.1, base_duration + duration_adjustment)
                
                word_end = min(current_time + word_duration, end_time)
                
                word_timings.append({
                    "word": word,
                    "start": round(current_time, 2),
                    "end": round(word_end, 2)
                })
                
                current_time = word_end
            
            logger.debug(f"📝 Estimated timing for {len(words)} words in {total_duration:.1f}s segment")
            return word_timings
            
        except Exception as e:
            logger.error(f"❌ Failed to estimate word timing: {str(e)}")
            return []
    
    def create_caption_chunks(self, text: str, start_time: float, end_time: float, speaker: str) -> List[Dict]:
        """
        Break text into caption-sized chunks with timing
        
        Args:
            text: The text to chunk
            start_time: Segment start time  
            end_time: Segment end time
            speaker: Speaker name (for styling)
            
        Returns:
            List of caption chunks: [{"text": str, "start": float, "end": float, "speaker": str}, ...]
        """
        try:
            # Get word-level timing
            word_timings = self.estimate_word_timing(text, start_time, end_time)
            if not word_timings:
                return []
            
            caption_chunks = []
            current_chunk_words = []
            current_chunk_chars = 0
            chunk_start_time = start_time
            
            for i, word_timing in enumerate(word_timings):
                word = word_timing["word"]
                word_start = word_timing["start"]
                word_end = word_timing["end"]
                
                # Check if adding this word would exceed character limit
                word_length = len(word) + (1 if current_chunk_words else 0)  # +1 for space
                
                if (current_chunk_chars + word_length > self.max_chars_per_line and current_chunk_words) or \
                   (word_end - chunk_start_time > self.max_duration_per_caption and current_chunk_words):
                    
                    # Create chunk from current words
                    if current_chunk_words:
                        chunk_text = " ".join(current_chunk_words)
                        chunk_end_time = word_timings[i-1]["end"] if i > 0 else word_end
                        
                        caption_chunks.append({
                            "text": chunk_text,
                            "start": chunk_start_time,
                            "end": chunk_end_time,
                            "speaker": speaker
                        })
                        
                        # Start new chunk
                        current_chunk_words = [word]
                        current_chunk_chars = len(word)
                        chunk_start_time = word_start
                    else:
                        # Single word is too long, use it anyway
                        current_chunk_words.append(word)
                        current_chunk_chars += word_length
                else:
                    # Add word to current chunk
                    current_chunk_words.append(word)
                    current_chunk_chars += word_length
            
            # Add final chunk if any words remain
            if current_chunk_words:
                chunk_text = " ".join(current_chunk_words)
                caption_chunks.append({
                    "text": chunk_text,
                    "start": chunk_start_time,
                    "end": end_time,
                    "speaker": speaker
                })
            
            logger.debug(f"📝 Created {len(caption_chunks)} caption chunks from text: '{text[:30]}...'")
            return caption_chunks
            
        except Exception as e:
            logger.error(f"❌ Failed to create caption chunks: {str(e)}")
            return []
    
    def enhance_timeline_with_captions(self, timeline: List[Dict]) -> Dict[str, Any]:
        """
        Enhance existing timeline with word-level timing and caption chunks
        
        Args:
            timeline: Original timeline from create_speaker_timeline()
            
        Returns:
            Enhanced timeline with caption data:
            {
                "segments": [...],  # Original segments with word_timings added
                "captions": [...],  # All caption chunks in chronological order
                "total_duration": float,
                "speaker_stats": {...}
            }
        """
        try:
            logger.info(f"🎬 Enhancing timeline with caption data for {len(timeline)} segments")
            
            enhanced_segments = []
            all_captions = []
            total_duration = 0
            speaker_stats = {
                "elon": {"duration": 0, "words": 0}, 
                "trump": {"duration": 0, "words": 0},
                "baburao": {"duration": 0, "words": 0},
                "samay": {"duration": 0, "words": 0}
            }
            
            for segment in timeline:
                speaker = segment['speaker']
                start_time = segment['start_time'] 
                end_time = segment['end_time']
                text = segment['text']
                
                # Add word-level timing
                word_timings = self.estimate_word_timing(text, start_time, end_time)
                
                # Create caption chunks
                caption_chunks = self.create_caption_chunks(text, start_time, end_time, speaker)
                
                # Update enhanced segment
                enhanced_segment = {
                    **segment,  # Keep original data
                    "word_timings": word_timings,
                    "caption_chunks": caption_chunks,
                    "word_count": len(word_timings)
                }
                enhanced_segments.append(enhanced_segment)
                
                # Add to global caption list
                all_captions.extend(caption_chunks)
                
                # Update stats
                segment_duration = end_time - start_time
                speaker_stats[speaker]["duration"] += segment_duration
                speaker_stats[speaker]["words"] += len(word_timings)
                total_duration = max(total_duration, end_time)
            
            # Sort captions chronologically
            all_captions.sort(key=lambda x: x["start"])
            
            enhanced_timeline = {
                "segments": enhanced_segments,
                "captions": all_captions,
                "total_duration": total_duration,
                "speaker_stats": speaker_stats,
                "caption_count": len(all_captions)
            }
            
            logger.info(f"✅ Enhanced timeline created:")
            logger.info(f"   📊 {len(enhanced_segments)} segments")
            logger.info(f"   💬 {len(all_captions)} caption chunks")
            logger.info(f"   ⏱️ {total_duration:.1f}s total duration")
            logger.info(f"   🗣️ Elon: {speaker_stats['elon']['duration']:.1f}s ({speaker_stats['elon']['words']} words)")
            logger.info(f"   🗣️ Trump: {speaker_stats['trump']['duration']:.1f}s ({speaker_stats['trump']['words']} words)")
            
            return enhanced_timeline
            
        except Exception as e:
            logger.error(f"❌ Failed to enhance timeline with captions: {str(e)}")
            raise Exception(f"Failed to enhance timeline with captions: {str(e)}")
    
    def get_current_caption(self, current_time: float, captions: List[Dict]) -> Dict:
        """
        Get the caption that should be displayed at a specific time
        
        Args:
            current_time: Time in seconds
            captions: List of caption chunks
            
        Returns:
            Caption dict or None if no caption at this time
        """
        for caption in captions:
            if caption["start"] <= current_time <= caption["end"]:
                return caption
        return None

# Global instance
caption_processor = CaptionProcessor()

def enhance_timeline_with_captions(timeline: List[Dict]) -> Dict[str, Any]:
    """
    Main function to enhance timeline with caption data
    """
    return caption_processor.enhance_timeline_with_captions(timeline)

def get_current_caption(current_time: float, captions: List[Dict]) -> Dict:
    """
    Get current caption for a specific time
    """
    return caption_processor.get_current_caption(current_time, captions)