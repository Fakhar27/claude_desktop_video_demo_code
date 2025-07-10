"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { AppSidebar } from "../(main)/_components/AppSidebar";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { GlitchText } from "../(main)/_components/ui/glitch-text";
import { BookOpen, Sparkles, Clock, RefreshCw } from "lucide-react";

// Hardcoded genres as requested
const GENRES = [
  { value: "fantasy", label: "Fantasy" },
  { value: "sci-fi", label: "Science Fiction" },
  { value: "mystery", label: "Mystery" },
  { value: "romance", label: "Romance" },
  { value: "horror", label: "Horror" },
  { value: "adventure", label: "Adventure" },
  { value: "thriller", label: "Thriller" },
  { value: "drama", label: "Drama" },
  { value: "comedy", label: "Comedy" },
  { value: "historical", label: "Historical" },
  { value: "cyberpunk", label: "Cyberpunk" },
  { value: "steampunk", label: "Steampunk" },
];

interface StoryResponse {
  success: boolean;
  stories: string[];
  iterations: number;
  metrics: {
    total_tokens: number;
    successful_requests: number;
    failed_requests: number;
  };
}

export default function StoriesPage() {
  const [prompt, setPrompt] = useState("");
  const [genre, setGenre] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [stories, setStories] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [shouldSubmit, setShouldSubmit] = useState(false);

  // Check if submit button should be enabled
  const isSubmitEnabled = prompt.trim() && genre;

  // Handle form submission
  const handleSubmit = async () => {
    if (!isSubmitEnabled) return;
    setShouldSubmit(true);
  };

  // API call effect
  useEffect(() => {
    if (!shouldSubmit) return;

    const callAPI = async () => {
      setIsLoading(true);
      setError(null);
      setStories([]);

      try {
        const response = await fetch('/api/stories', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            prompt: prompt.trim(),
            genre: genre,
          }),
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.error || `HTTP Error: ${response.status}`);
        }

        const data: StoryResponse = await response.json();
        
        if (data.success && data.stories) {
          setStories(data.stories);
        } else {
          throw new Error('Invalid response format');
        }
      } catch (err) {
        console.error('Story generation error:', err);
        setError(err instanceof Error ? err.message : 'An unexpected error occurred');
      } finally {
        setIsLoading(false);
        setShouldSubmit(false);
      }
    };

    callAPI();
  }, [shouldSubmit, prompt, genre]);

  const handleReset = () => {
    setPrompt("");
    setGenre("");
    setStories([]);
    setError(null);
  };

  return (
    <div className="flex min-h-screen w-full bg-black">
      <AppSidebar />
      
      {/* Main Content */}
      <div className="flex-1 flex flex-col w-full md:pl-64">
        <div className="relative">
          <div className="absolute inset-0 z-0 opacity-30">
            <div className="absolute top-0 left-0 right-0 h-[500px] bg-gradient-to-b from-pink-600/20 via-purple-600/10 to-transparent"></div>
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,rgba(255,0,255,0.15),transparent_50%)]"></div>
          </div>
          
          <div className="relative w-full max-w-7xl mx-auto p-4 sm:p-6 z-10">
            {/* Header */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="text-center mb-8"
            >
              <GlitchText className="text-white text-2xl sm:text-3xl md:text-5xl font-bold tracking-tighter">
                AI Story Generator
              </GlitchText>
              <div className="h-1 w-24 sm:w-32 md:w-48 bg-gradient-to-r from-purple-500 via-pink-500 to-red-500 mx-auto mt-2" />
              <p className="text-zinc-400 mt-4 max-w-2xl mx-auto text-sm sm:text-base">
                Generate synchronized sequential stories with AI. Enter your prompt and select a genre to create four connected story segments.
              </p>
            </motion.div>

            {/* Error Display */}
            {error && (
              <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                className="mb-6 p-4 bg-red-900/50 border border-red-800 rounded-lg text-white max-w-4xl mx-auto"
              >
                <p className="font-bold text-red-300">Error:</p>
                <p>{error}</p>
                <Button
                  variant="outline"
                  onClick={() => setError(null)}
                  className="mt-2 text-sm border-red-700 text-red-300 hover:bg-red-800/50"
                >
                  Dismiss
                </Button>
              </motion.div>
            )}

            {/* Input Form */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="max-w-4xl mx-auto mb-8"
            >
              <Card className="bg-zinc-900/50 border-zinc-800 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="text-white flex items-center gap-2">
                    <BookOpen className="h-5 w-5 text-pink-500" />
                    Story Configuration
                  </CardTitle>
                  <CardDescription className="text-zinc-400">
                    Enter your story prompt and select a genre to generate four sequential story segments.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  {/* Prompt Input */}
                  <div className="space-y-2">
                    <Label htmlFor="prompt" className="text-white font-medium">
                      Story Prompt
                    </Label>
                    <Input
                      id="prompt"
                      type="text"
                      placeholder="Enter your story idea... (e.g., 'A brave knight embarks on a quest to find the lost treasure')"
                      value={prompt}
                      onChange={(e) => setPrompt(e.target.value)}
                      className="bg-zinc-800/50 border-zinc-700 text-white placeholder-zinc-500 focus:border-pink-500 focus:ring-pink-500/20"
                      disabled={isLoading}
                    />
                  </div>

                  {/* Genre Selection */}
                  <div className="space-y-2">
                    <Label htmlFor="genre" className="text-white font-medium">
                      Genre
                    </Label>
                    <Select value={genre} onValueChange={setGenre} disabled={isLoading}>
                      <SelectTrigger className="bg-zinc-800/50 border-zinc-700 text-white focus:border-pink-500 focus:ring-pink-500/20">
                        <SelectValue placeholder="Select a genre..." />
                      </SelectTrigger>
                      <SelectContent className="bg-zinc-800 border-zinc-700">
                        {GENRES.map((genreOption) => (
                          <SelectItem 
                            key={genreOption.value} 
                            value={genreOption.value}
                            className="text-white hover:bg-zinc-700 focus:bg-zinc-700"
                          >
                            {genreOption.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  {/* Submit Button */}
                  <div className="flex flex-col sm:flex-row gap-3 pt-4">
                    <Button
                      onClick={handleSubmit}
                      disabled={!isSubmitEnabled || isLoading}
                      className="flex-1 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:from-zinc-700 disabled:to-zinc-600 text-white font-medium"
                    >
                      {isLoading ? (
                        <>
                          <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                          Generating Stories...
                        </>
                      ) : (
                        <>
                          <Sparkles className="mr-2 h-4 w-4" />
                          Generate Stories
                        </>
                      )}
                    </Button>
                    
                    {stories.length > 0 && !isLoading && (
                      <Button
                        variant="outline"
                        onClick={handleReset}
                        className="border-zinc-600 text-zinc-300 hover:bg-zinc-800"
                      >
                        Reset
                      </Button>
                    )}
                  </div>
                </CardContent>
              </Card>
            </motion.div>

            {/* Stories Display */}
            <AnimatePresence>
              {stories.length > 0 && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.5 }}
                  className="max-w-4xl mx-auto"
                >
                  <Card className="bg-zinc-900/50 border-zinc-800 backdrop-blur-sm">
                    <CardHeader>
                      <CardTitle className="text-white flex items-center gap-2">
                        <BookOpen className="h-5 w-5 text-green-500" />
                        Generated Stories
                      </CardTitle>
                      <CardDescription className="text-zinc-400">
                        Four sequential story segments based on your prompt "{prompt}" in the {GENRES.find(g => g.value === genre)?.label} genre.
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="grid gap-4">
                        {stories.map((story, index) => (
                          <motion.div
                            key={index}
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ duration: 0.3, delay: index * 0.1 }}
                            className="relative"
                          >
                            <Card className="bg-zinc-800/50 border-zinc-700 hover:border-zinc-600 transition-colors">
                              <CardHeader className="pb-3">
                                <div className="flex items-center gap-2">
                                  <div className="w-8 h-8 rounded-full bg-gradient-to-r from-purple-500 to-pink-500 flex items-center justify-center">
                                    <span className="text-white font-bold text-sm">{index + 1}</span>
                                  </div>
                                  <CardTitle className="text-white text-lg">
                                    Story Segment {index + 1}
                                  </CardTitle>
                                </div>
                              </CardHeader>
                              <CardContent>
                                <p className="text-zinc-300 leading-relaxed">
                                  {story}
                                </p>
                              </CardContent>
                            </Card>
                          </motion.div>
                        ))}
                      </div>
                      
                      {/* Story Stats */}
                      <div className="mt-6 p-4 bg-zinc-800/30 rounded-lg border border-zinc-700">
                        <div className="flex items-center gap-4 text-sm text-zinc-400">
                          <div className="flex items-center gap-1">
                            <Clock className="h-4 w-4" />
                            <span>4 Iterations</span>
                          </div>
                          <div className="flex items-center gap-1">
                            <BookOpen className="h-4 w-4" />
                            <span>{stories.length} Stories Generated</span>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Loading State */}
            {isLoading && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="max-w-4xl mx-auto"
              >
                <Card className="bg-zinc-900/50 border-zinc-800 backdrop-blur-sm">
                  <CardContent className="py-12">
                    <div className="text-center">
                      <RefreshCw className="h-12 w-12 mx-auto mb-4 text-pink-500 animate-spin" />
                      <h3 className="text-xl font-bold text-white mb-2">
                        Generating Your Stories
                      </h3>
                      <p className="text-zinc-400 mb-4">
                        Creating 4 sequential story segments...
                      </p>
                      <div className="flex justify-center gap-1">
                        {[...Array(4)].map((_, i) => (
                          <div
                            key={i}
                            className="w-2 h-2 bg-pink-500 rounded-full animate-pulse"
                            style={{
                              animationDelay: `${i * 0.2}s`,
                            }}
                          />
                        ))}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
