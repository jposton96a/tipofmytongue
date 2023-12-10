###### Build Stage

FROM node:alpine AS builder
WORKDIR /app

# Install app dependencies
COPY package*.json ./
RUN npm install

# Copy app source to the container
COPY tsconfig.json ./
COPY public ./public
COPY src ./src

# Build the app
RUN npm run build

####### Nginx Server

FROM nginx:alpine
# Set working directory to nginx asset directory
WORKDIR /usr/share/nginx/html
# Remove default nginx static assets
RUN rm -rf ./*
# Copy static assets from builder stage
COPY --from=builder /app/build/ ./

# COPY nginx/default.conf /etc/nginx/conf.d/default.conf
COPY nginx/default.conf.template /etc/nginx/conf.d/default.conf.template
ENV API_SERVER "localhost:8100"

# Containers run nginx with global directives and daemon off
# ENTRYPOINT ["nginx", "-g", "daemon off;"]
ENTRYPOINT ["/bin/sh", "-c", "envsubst < /etc/nginx/conf.d/default.conf.template > /etc/nginx/conf.d/default.conf && nginx -g 'daemon off;'"]
# ENTRYPOINT ["/bin/sh"]