from django.contrib.auth.models import User
from rest_framework.decorators import api_view,permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
import asyncio
import json
import traceback
from rest_framework import status
from rest_framework_simplejwt.views import TokenObtainPairView
from .models import notes
from django.http import JsonResponse
from django.contrib.auth.models import User
from django.views.decorators.csrf import csrf_exempt
# from .services.langchain_service import generate_content_pipeline
from .services.langchain_service import ContentRequest, StoryIterationChain
from .serializers import notesSerializers
import requests
from asgiref.sync import async_to_sync
from django.http import JsonResponse
import base64
import io
import cohere
import logging
from dotenv import load_dotenv 
import os
from pydantic import ValidationError

# ------------------------ LOGGING ------------------------ #
logger = logging.getLogger(__name__)
load_dotenv()

# ------------------------ APIs ------------------------ #
co = cohere.Client(os.getenv("CO_API_KEY"))
# API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
# API_URL = "https://api-inference.huggingface.co/models/Lykon/DreamShaper"
# headers = {"Authorization": os.getenv(HUGGING_FACE_API_KEY)}
COLAB_URL = "https://87c7-35-185-226-172.ngrok-free.app"
COLAB_URL_2 = ""
COLAB_URL_3 = ""
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# if not OPENAI_API_KEY:
#     logger.warning("OPENAI_API_KEY environment variable not set")


# ------------------------ FUNCTIONS ------------------------ #

# use with colab or sagemaker
@csrf_exempt
def update_ngrok_url(request):
    global COLAB_URL  
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            ngrok_url = data.get('ngrok_url')
            if not ngrok_url:
                return JsonResponse({"error": "Ngrok URL is required"}, status=400)

            COLAB_URL = ngrok_url
            print(f"Received and updated Ngrok URL: {COLAB_URL}")

            return JsonResponse({"message": "Ngrok URL updated successfully"}, status=200)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
    else:
        return JsonResponse({"error": "Invalid request method"}, status=405)
    
@csrf_exempt
def update_ngrok_url_voice(request):
    """Endpoint for Colab to register its URL"""
    global COLAB_URL_2
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            ngrok_url = data.get('ngrok_url')
            if ngrok_url:
                COLAB_URL_2 = ngrok_url
                return JsonResponse({"message": "URL updated successfully"})
            return JsonResponse({"error": "No URL provided"}, status=400)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
    return JsonResponse({"error": "Invalid method"}, status=405)

@csrf_exempt
def update_ngrok_url_whisper(request):
    """ENDPOINT FOR NGROK COLAB NOTEBOOK FOR WHISPER SOUND PROCESSING"""
    global COLAB_URL_3
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            ngrok_url = data.get("ngrok_url")
            if ngrok_url:
                COLAB_URL_3 = ngrok_url
                return JsonResponse({"message": "URL updated successfully"})
            return JsonResponse({"error": "No URL provided"}, status=400)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
    return JsonResponse({"error": "Invalid method"}, status=405)
    
langchain_service = None
story_chain_service = None

async def get_story_chain_service():
    """Async factory function for StoryIterationChain"""
    try:
        service = StoryIterationChain(
            # colab_url=COLAB_URL,
            colab_url="abcd", 
            voice_url=COLAB_URL_2, 
            whisper_url=COLAB_URL_3
        )
        logger.info("StoryIterationChain service created successfully")
        return service
    except Exception as e:
        logger.error(f"Error creating StoryIterationChain service: {str(e)}")
        raise
    
async def get_story_chain_service_2():
    """Async factory function for StoryIterationChain"""
    try:
        service = StoryIterationChain(
            voice_url=COLAB_URL_2, 
            whisper_url=COLAB_URL_3
        )
        logger.info("StoryIterationChain service created successfully")
        return service
    except Exception as e:
        logger.error(f"Error creating StoryIterationChain service: {str(e)}")
        raise
    
@csrf_exempt
async def wan_video_generation_request(request):
    try:
        data = json.loads(request)
        
        prompt = data.get("prompt")
        genre = "cinematic"
        negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        iterations = 3
        guidance_scale = 5
        
        content_request = ContentRequest(
            prompt=prompt,
            genre=genre,
            iterations=iterations,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale
        )
        
        logger.info(f"Content request: {content_request}")
        
        story = StoryIterationChain()
        result = await story.generate_video_WAN_semi(content_request)
        
        response_data = {
            "success": True,
            "video_data": result["video_data"],
            "content_type": result["content_type"],
            "metrics": result["metrics"]
        }
        
        logger.info("Returning video response")
        return JsonResponse(response_data, status=200)
    except Exception as e:
        error_msg = f"Content generation error: {str(e)}"
        logger.error(error_msg)
        return JsonResponse({"error": error_msg}, status=500)
    
@csrf_exempt
async def wan_video_generation_request_complete_pipeline(request):
    try:
        data = json.loads(request)
        
        prompt = data.get("prompt")
        genre = "cinematic"
        negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        iterations = 3
        guidance_scale = 5
        
        content_request = ContentRequest(
            prompt=prompt,
            genre=genre,
            iterations=iterations,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            backgroundVideo="0",
            backgroundMusic="0",
            voiceType="v2/en_speaker_6",
            subtitleColor="#ff00ff"
        )
        
        logger.info(f"Content request: {content_request}")
        
        service = await get_story_chain_service_2()
        result = await service.generate_video_WAN(content_request)
        
        response_data = {
            "success": True,
            "video_data": result["video_data"],
            "content_type": result["content_type"],
            "metrics": result["metrics"]
        }
        
        logger.info("Returning video response")
        return JsonResponse(response_data, status=200)
    except Exception as e:
        error_msg = f"Content generation error: {str(e)}"
        logger.error(error_msg)
        return JsonResponse({"error": error_msg}, status=500)
        
@csrf_exempt
async def generate_content(request):
    if request.method != 'POST':
        return JsonResponse({"error": "Invalid request method."}, status=405)
        
    try:
        data = json.loads(request.body)
        
        if not COLAB_URL or not COLAB_URL_2 or not COLAB_URL_3:
            logger.error("COLAB_URL or COLAB_URL_2 or COLAB_URL_3 not set")
            return JsonResponse({
                "error": "Required services not configured. Update URLs first."
            }, status=500)
            
        # content_request = ContentRequest(
        #     prompt=data.get("prompt"),
        #     genre=data.get("genre", "Adventure"),
        #     iterations=data.get("iterations", 4)
        # )
        content_request = ContentRequest(
            prompt=data.get("prompt"),
            genre=data.get("genre", "cyberpunk"),
            iterations=data.get("iterations", 4),
            backgroundVideo=data.get("backgroundType", "urban"),
            backgroundMusic=data.get("musicType", "synthwave"),
            voiceType=data.get("voiceType", "male"),
            subtitleColor=data.get("subtitleColor", "#ff00ff")
        )
        
        logger.info(f"Content request: {content_request}")
        
        service = await get_story_chain_service()
        result = await service.generate_content_pipeline(content_request)
        
        response_data = {
            "success": True,
            "video_data": result["video_data"],
            "content_type": result["content_type"],
            "metrics": result["metrics"]
        }
        
        logger.info("Returning video response")
        return JsonResponse(response_data, status=200)
            
    except Exception as e:
        error_msg = f"Content generation error: {str(e)}"
        logger.error(error_msg)
        return JsonResponse({"error": error_msg}, status=500)
# @csrf_exempt
# async def generate_content(request):
#     if request.method != 'POST':
#         return JsonResponse({"error": "Invalid request method."}, status=405)
        
#     try:
#         data = json.loads(request.body)
        
#         # TESTING ONLY: Override parameters for testing
#         test_mode = "normal"  # Options: "normal", "hf_image", "hf_video"
        
#         # Force Hugging Face for testing
#         if test_mode == "hf_image":
#             logger.info("TESTING MODE: Using Hugging Face for image generation")
#             use_hf_inference = True
#             use_hf_video = False
#         elif test_mode == "hf_video":
#             logger.info("TESTING MODE: Using Hugging Face for direct video generation")
#             use_hf_inference = True
#             use_hf_video = True
#         else:
#             use_hf_inference = not COLAB_URL
#             use_hf_video = False
            
#         content_request = ContentRequest(
#             prompt=data.get("prompt"),
#             genre=data.get("genre", "cyberpunk"),
#             iterations=data.get("iterations", 4),
#             backgroundVideo=data.get("backgroundType", "urban"),
#             backgroundMusic=data.get("musicType", "synthwave"),
#             voiceType=data.get("voiceType", "v2/en_speaker_6"),
#             subtitleColor=data.get("subtitleColor", "#ff00ff"),
#             useHfInference=use_hf_inference,
#             useHfVideo=use_hf_video,
#             hfImageModel="black-forest-labs/FLUX.1-schnell",
#             hfVideoModel="Wan-AI/Wan2.1-T2V-14B"
#         )
        
#         logger.info(f"Content request: {content_request}")
        
#         service = await get_story_chain_service()
#         result = await service.generate_content_pipeline(content_request)
        
#         response_data = {
#             "success": True,
#             "video_data": result["video_data"],
#             "content_type": result["content_type"],
#             "metrics": result["metrics"]
#         }
        
#         logger.info("Returning video response")
#         return JsonResponse(response_data, status=200)
            
#     except Exception as e:
#         error_msg = f"Content generation error: {str(e)}"
#         logger.error(error_msg)
#         logger.error(traceback.format_exc())  # Add traceback for better debugging
#         return JsonResponse({"error": error_msg}, status=500)
    

@csrf_exempt
def generate_voice(request):
    """Handle voice generation requests"""
    global COLAB_URL_2
    if request.method == 'POST':
        try:
            if not COLAB_URL_2:
                return JsonResponse({"error": "Colab service not available"}, status=503)

            data = json.loads(request.body)
            text = data.get('text')
            
            if not text:
                return JsonResponse({"error": "Text is required"}, status=400)

            response = requests.post(
                f"{COLAB_URL_2}/generate-speech",
                json={"text": text},
                timeout=90
            )
            
            if response.status_code == 200:
                return JsonResponse(response.json())
            else:
                return JsonResponse(
                    {"error": "Failed to generate audio"}, 
                    status=response.status_code
                )

        except requests.exceptions.RequestException as e:
            return JsonResponse(
                {"error": f"Failed to connect to Colab service: {str(e)}"}, 
                status=503
            )
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid method"}, status=405)
        
    
    


# ---------------------- AUTH AND USER MANAGEMENT ----------------------
class MyTokenObtainPairSerializer(TokenObtainPairSerializer):
    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)
        # Add custom claims
        token['username'] = user.username
        token['password'] = user.password
        # ...
        return token

class MyTokenObtainPairView(TokenObtainPairView):
    serializer_class = MyTokenObtainPairSerializer

@api_view(['GET'])
def getRoutes(request):
    routes = [
        '/api/token',
        '/api/token/refresh',
    ]
    return Response(routes)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def getUserDetails(request):
    user = request.user
    user_data = {
        'id': user.id,
        'username': user.username,
    }
    return Response(user_data)

@api_view(['POST'])
def create(request):
    data = request.data
    username = data.get("username", "").lower()
    password = data.get("password", "")
    if User.objects.filter(username=username).exists():
        return Response({"error": "USER ALREADY EXISTS"}, status=status.HTTP_400_BAD_REQUEST)
    try:
        user = User.objects.create_user(username=username,password=password)
        user.save()
        return Response({"message": "User created successfully"}, status=status.HTTP_201_CREATED)
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def getNotes(request):
    user = request.user
    Notes = user.notes_set.all()  
    serializer = notesSerializers(Notes, many=True)
    return Response(serializer.data)










# def generate_image(request):
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)
#             prompt = data.get('prompt')
#             logger.info(f"Received prompt: {prompt}")
#         except json.JSONDecodeError as e:
#             logger.error(f"JSON decode error: {str(e)}")
#             return JsonResponse({"error": "Invalid JSON"}, status=400)

#         if not prompt:
#             logger.error("No prompt provided")
#             return JsonResponse({"error": "Prompt is required"}, status=400)

#         try:
#             logger.info(f"Sending request to Hugging Face API with prompt: {prompt}")
#             response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
#             response.raise_for_status()

#             image_bytes = response.content
#             image_base64 = base64.b64encode(image_bytes).decode('utf-8')
#             logger.info("Successfully generated image")
#             return JsonResponse({"image_data": image_base64}, status=200)
#         except requests.exceptions.RequestException as e:
#             logger.error(f"Error from Hugging Face API: {str(e)}")
#             return JsonResponse({"error": f"Error from Hugging Face API: {str(e)}"}, status=500)
#         except Exception as e:
#             logger.error(f"Unexpected error: {str(e)}")
#             return JsonResponse({"error": f"Unexpected error: {str(e)}"}, status=500)