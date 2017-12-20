classdef Circuit
    %CIRCUIT Summary of this class goes here
    %   Detailed explanation goes here

    properties

        img_circuit;
        img_best_path;
        one_normalized_path;
        road_location;

        width;
        height;

        start;
        goal;
    end

    methods

        % ----- Constructor
        function obj = Circuit(image_path)

            % Load image of the circuit
            obj.img_circuit = imread(image_path);

            % Set circuit dimensions
            obj.height = size(obj.img_circuit,1);
            obj.width = size(obj.img_circuit,2);

            % Compute best path matrix using distance transform
            bw_circuit_img = im2bw(obj.img_circuit, 0.1); %thresh
            obj.img_best_path = bwdist(bw_circuit_img); % distance transform
            obj.img_best_path(obj.img_best_path==0) = - max(max(obj.img_best_path));

            obj.one_normalized_path = obj.img_best_path(:) ./ max(max(obj.img_best_path));

            obj.road_location = find(any(obj.img_circuit(:,:,:), 3)==0);
        end

        % ----- Display circuit
        function im = display_circuit(obj)
            im = image(obj.img_circuit);
        end

        % ----- Display best path on the circuit
        function im = display_best_path(obj)
            im = imagesc(obj.img_best_path);
        end

        % ----- Get reward given car location, considering best path
        function [reward_c,reward_g, reward_v] = calculate_reward(obj, car, car_old_location, car_new_location)

            reward = 0;

            x_new = car_new_location(1);
            y_new = car_new_location(2);

            if(x_new > 0 && x_new < size(obj.img_best_path,2))
                if(y_new > 0 && y_new < size(obj.img_best_path,1))
                    reward = obj.img_best_path(y_new,x_new);
                end
            end

            % Reward is zero if location isn't changed
            if(all(car_old_location == car_new_location))
                reward = 0;
            end

            reward_c=reward;%/max(max(obj.img_best_path));
%             % Goal reward
%             reward_g=1-(pdist([car_new_location' ;obj.goal'])/pdist([obj.goal' ; obj.start']));
%
%             if(all(car_new_location==obj.goal))
%                 reward_g=1;
%             end
            reward_g = 0;

            % reward velocity
            v_x = car.velocity * cos(car.angle);
            v_y = car.velocity * sin(car.angle);

            mask = zeros(size(obj.img_best_path));
            x = car.location(1);
            y = car.location(2);
            for i=0:10
                mask(round(y + v_y*i/10), round(x+v_x*i/10)) = 1;
            end

            reward_v = obj.one_normalized_path'*mask(:); % = sum(one_normalized_path(mask==1));
%             imagesc(one_normalized_path);hold on;plot(x, y, 'ro');hold off;
%             figure;imagesc(mask);hold on;plot(x, y, 'ro');hold off;
        end

        function ok = car_in_borders (obj, car_location)

            ok = false;

            x = car_location(1);
            y = car_location(2);

            if(x > 0 && ...
                    x < size(obj.img_best_path,2) && ...
                    y > 0 && ...
                    y < size(obj.img_best_path,1) )
                ok = true;
            end

        end


    end

end