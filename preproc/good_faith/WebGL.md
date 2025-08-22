
bindBuffer() - Tells WebGL "I want to work with this specific buffer"
bufferData() - Uploads the JavaScript array to GPU memory with DYNAMIC_DRAW (meaning the data changes frequently)

Performance Benefits to buffers:
- GPU Memory Speed: Buffers store data directly in GPU memory (VRAM), which is much faster to access than sending data from JavaScript/CPU each frame
- Batch Processing: The GPU can process thousands of points in parallel once the data is in buffers, rather than handling them one-by-one
- Reduced Overhead: Without buffers, you'd need to send position/color/size data across the CPU-GPU bridge every frame, creating a major bottleneck

Memory Efficiency to buffers:
- Single Upload: Data gets uploaded once to GPU memory and stays there until you need to change it
- No Repeated Transfers: Instead of sending arrays every render call, you just tell the GPU "use the data that's already there"

So buffers are good for interactivity

When you'd use a buffer:
- For interactive applications (like this tweet visualization):
- Buffers are very beneficial because you're rendering the same data many times at 60fps
- User can zoom, pan, hover, etc. - all using the same buffer data

For static snapshots (render once and save) a buffer is less useful:
 - The benefit is much less because you're only rendering each dataset once
 - You don't get the "upload once, render many times" advantage
