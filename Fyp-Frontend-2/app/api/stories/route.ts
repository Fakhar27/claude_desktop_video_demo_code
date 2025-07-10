// app/api/stories/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { auth } from "@clerk/nextjs/server";

// Get backend URL with robust fallback strategy
const BACKEND_URL = process.env.BACKEND_URL || 
  (process.env.NODE_ENV === 'production' 
    ? 'http://backend:8000'  // Docker service name for ECS internal communication
    : 'http://localhost:8000'); // Local development

export async function POST(request: NextRequest) {
  try {
    // Get the current authenticated user
    const { userId } = await auth();
    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const body = await request.json();
    const { prompt, genre } = body;

    // Validate required fields
    if (!prompt || !genre) {
      return NextResponse.json({ 
        error: 'Both prompt and genre are required' 
      }, { status: 400 });
    }

    console.log(`Attempting to connect to backend at http://127.0.0.1:8000/story-completion/`);

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5 * 60 * 1000); 

      const response = await fetch(`http://127.0.0.1:8000/story-completion/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt: prompt.trim(),
          genre: genre,
        }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Backend error response:', errorText);
        
        let errorData;
        try {
          errorData = JSON.parse(errorText);
        } catch {
          errorData = { error: errorText };
        }

        return NextResponse.json(
          {
            error: 'Story generation failed',
            details: errorData.error || 'Unknown backend error',
          },
          { status: response.status }
        );
      }

      const responseData = await response.json();
      
      if (!responseData.success) {
        return NextResponse.json({ 
          error: 'Story generation failed',
          details: responseData.error || 'Unknown error from backend'
        }, { status: 500 });
      }

      // Return the stories data
      return NextResponse.json({
        success: true,
        stories: responseData.stories,
        iterations: responseData.iterations,
        metrics: responseData.metrics
      });

    } catch (error: any) {
      console.error('Backend fetch error:', error);

      if (error.name === 'AbortError') {
        return NextResponse.json(
          {
            error: 'Request timed out after 5 minutes. Story generation is taking too long.',
          },
          { status: 504 }
        );
      }

      return NextResponse.json(
        {
          error: 'Failed to connect to story generation service',
          details: error.message || 'Unknown error',
        },
        { status: 502 }
      );
    }
  } catch (error: any) {
    console.error('API route error:', error);
    return NextResponse.json(
      {
        error: 'Error processing request',
        message: error.message,
      },
      { status: 500 }
    );
  }
}
