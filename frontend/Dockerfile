FROM node:16-alpine

# Expose ports
EXPOSE 5049
EXPOSE 3000
EXPOSE 35729

# Set working directory
WORKDIR /app

# Add /app/node_modules/.bin to environment variables
ENV PATH /app/node_modules/.bin:$PATH

# Copy package files and install app dependencies
COPY package.json /app/package.json
COPY package-lock.json /app/package-lock.json
RUN npm install typescript --save-dev
RUN npm install
RUN npm install react-scripts -g

#TEST
RUN mkdir -p node_modules/.cache && chmod -R 777 node_modules/.cache

# Add React app to working directory
ADD . /app

#TEST
RUN chmod 777 $(find /app/node_modules/. -type d)
USER node

# Start the React app
CMD ["npm", "start"]